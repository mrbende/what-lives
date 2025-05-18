import asyncio
import json
import logging
import os
import re
import time

import boto3
import botocore
import tiktoken
import yaml
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AsyncAnthropicBedrock,
    APIError,
    APIStatusError,
    AsyncAnthropic,
    RateLimitError,
)
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from IPython.display import Markdown, clear_output, display, display_markdown
from openai import AsyncOpenAI, OpenAI
import uuid

class Inference:
    def __init__(self):
        ### LOAD ENVIRONMENT VARIABLES
        load_dotenv()

        ### REROUTE ANTHROPIC API --> BEDROCK ASYNC ENDPOINT
        self.redirect_anthropic_to_bedrock = False
        # ! need to update model_config if False !
        
        ### RUN CONFIG
        self._system_init()
        self._hyperparameter_init()
        self._api_client_init()
        self._logging_init()
        self._embedding_init()
        self._model_config_init()
    

    #####################
    ### CONFIGURATION ###
    #####################
    def _system_init(self):
        ### SYSTEM PATHS
        self.prompt_template_location = "/workspace/what-lives/prompts/"

        ### API RETRY PARAMETERS WITH EXPONENTIAL BACKOFF
        self.max_retries = 3
        self.initial_wait = 1.0
        self.max_wait = 32.0

    def _hyperparameter_init(self):
        self.temperature = 0.3  # flatness / steepness of probability distribution -- higher is more creative
        self.top_k = 200  # only sample from the tok_k options for each subsequent token -- higher is more creative
        self.top_p = (
            0.9  # cuts off the probability distribution -- <1 is more deterministic
        )

    def _api_client_init(self):
        timer = Timer()
        authenticated_clients = []
        missing_clients = []

        ### OPENAI INIT
        if os.environ.get("OPENAI_API_KEY"):
            try:
                self.openai_client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
                self.openai_aclient = AsyncOpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
                authenticated_clients.append("OpenAI")
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
                missing_clients.append("OpenAI")
        else:
            print("Warning: `OPENAI_API_KEY` environment variable not found. OpenAI client will not be available.")
            missing_clients.append("OpenAI")
            self.openai_client = None
            self.openai_aclient = None

        ### ANTHROPIC INIT
        if not self.redirect_anthropic_to_bedrock:
            if os.environ.get("ANTHROPIC_API_KEY"):
                try:
                    self.anthropic_aclient = AsyncAnthropic(
                        api_key=os.environ.get("ANTHROPIC_API_KEY"),
                    )
                    authenticated_clients.append("Anthropic")
                except Exception as e:
                    print(f"Warning: Failed to initialize Anthropic client: {e}")
                    missing_clients.append("Anthropic")
                    self.anthropic_aclient = None
            else:
                print("Warning: `ANTHROPIC_API_KEY` environment variable not found. Anthropic client will not be available.")
                missing_clients.append("Anthropic")
                self.anthropic_aclient = None
        else:
            # Handle Anthropic via Bedrock
            aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

            if aws_access_key and aws_secret_key:
                try:
                    self.anthropic_aclient = AsyncAnthropicBedrock(
                        aws_access_key=aws_access_key,
                        aws_secret_key=aws_secret_key,
                        aws_region="us-east-1",
                    )
                    authenticated_clients.append("Anthropic via Bedrock")
                except Exception as e:
                    print(f"Warning: Failed to initialize Anthropic via Bedrock: {e}")
                    missing_clients.append("Anthropic via Bedrock")
                    self.anthropic_aclient = None
            else:
                print("Warning: AWS credentials not found. Anthropic via Bedrock will not be available.")
                missing_clients.append("Anthropic via Bedrock")
                self.anthropic_aclient = None

        ### BEDROCK INIT
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        if aws_access_key and aws_secret_key:
            try:
                self.bedrock_client = boto3.client(
                    "bedrock-runtime",
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name="us-east-1",
                )
                self.bedrock_base_client = boto3.client(
                    "bedrock",
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name="us-east-1",
                )
                authenticated_clients.append("AWS Bedrock")
            except Exception as e:
                print(f"Warning: Failed to initialize AWS Bedrock clients: {e}")
                missing_clients.append("AWS Bedrock")
                self.bedrock_client = None
                self.bedrock_base_client = None
        else:
            print("Warning: AWS credentials not found. Bedrock clients will not be available.")
            missing_clients.append("AWS Bedrock")
            self.bedrock_client = None
            self.bedrock_base_client = None

        time_elapsed = timer.get_time()

        if authenticated_clients:
            print(f"Successfully authenticated to: {', '.join(authenticated_clients)} in {time_elapsed} seconds.")

        if missing_clients:
            print(f"WARNING: The following clients could not be authenticated: {', '.join(missing_clients)}")
            print("Some functionality may be limited.")

    def _logging_init(self):
        ### LOGGING FOR DATABASE OPERATIONS
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def _embedding_init(self):
        ### OPENAI CLIENT FOR EMBEDDINGS
        self.openai_embedding_model = "text-embedding-3-large"
        self.openai_embedding_dimensions = 1024

        ### BEDROCK CLIENT FOR EMBEDDINGS
        self.bedrock_embedding_model = 	"amazon.titan-embed-text-v2:0"
        self.bedrock_embedding_dimensions = 1024 # 1536  # specific to titan-embed-text-v1

        ### SET TOKENIZER FOR TOKEN / COST ACCOUNTING
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _model_config_init(self):
        ### LOAD MODEL CONFIG FROM YAML
        model_config = "/workspace/what-lives/src/model_config.yml"
        with open(model_config, "r") as f:
            self.model_config = yaml.safe_load(f)

        ### PARSE FOR LISTS OF MODELS AVAILABLE
        self.all_models = []
        self.agent_models = []
        for service, models in self.model_config["models"].items():
            ### LOOP THROUGH EACH MODEL IN SERVICE
            for model_name, details in models.items():
                self.all_models.append(model_name)

    def _get_model_meta(self, model_name):
        ### LOOP THROUGH MODEL CONFIG TO RETURN MODEL METADATA
        for service, models in self.model_config["models"].items():
            if model_name in models:
                return {
                    "name": model_name,
                    "service": service,
                    **models[model_name],  # Unpack all model details
                }
        ### IF NONE FOUND, RETURN NONE
        return None

    ########################################
    ### WRAPPING ALL INFERENCE FUNCTIONS ###
    ########################################
    async def acomplete(
        self, text, system_prompt=None, model=None, verbose=False, numerical=False
    ):
        ### SET MODEL
        model = (
            model if model else self.model_config["default_models"]["global_default"]
        )
        if model not in self.all_models:
            raise ValueError("Invalid model name provided.")
        model_meta = self._get_model_meta(model)

        ### CALL PROPER SERVICE FOR ASYNC INFERENCE
        if model_meta["service"] == "bedrock":
            response, metadata = await self.bedrock_acomplete(
                text, system_prompt, model=model, verbose=verbose, numerical=numerical
            )
        elif model_meta["service"] == "anthropic":
            response, metadata = await self.anthropic_acomplete(
                text, system_prompt, model=model, verbose=verbose, numerical=numerical
            )
        elif model_meta["service"] == "openai":
            response, metadata = await self.openai_acomplete(
                text, system_prompt, model=model, verbose=verbose, numerical=numerical
            )
        else:
            raise ValueError(
                f"Model service {model_meta["service"]} is not recognized."
            )

        ### RETURN COMPLETION
        return response, metadata

    ###########################
    ### ANTHROPIC INFERENCE ###
    ###########################
    ### ASYNCHRONOUS ONE-SHOT WITH RETRY
    async def anthropic_acomplete(
        self, text, system_prompt=None, model=None, verbose=True, numerical=False
    ):
        timer = Timer()

        ### SET MODEL
        model = model if model else self.model_config["default_models"]["anthropic"]
        if model not in self.model_config["models"]["anthropic"].keys():
            raise ValueError("Invalid Anthropic model name provided.")
        model_meta = self._get_model_meta(model)

        ### ALLOW FOR ASYNC RETRY ON API ERRORS
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                ### GENERATE MESSAGE THREAD
                user_message = self._form_message(text)

                ### CREATE SYSTEM PROMPT
                system_template = (
                    system_prompt
                    if system_prompt
                    else self._read_prompt_template("default_prompt")
                )

                ### ANTHROPIC API COMPLETION
                response = await self.anthropic_aclient.messages.create(
                    max_tokens=model_meta["max_tokens"],
                    system=system_template,
                    messages=user_message,
                    model=model_meta["model_id"],
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                response_text = response.content[0].text

                ### VALIDATE SQL ID NECESSARY
                if numerical:
                    try:
                        response_text = self._validate_float(response_text)
                    except Exception as e:
                        raise ValueError(f"Invalid float generated: {response_text}")

                ### COMPILE METADATA AND RETURN
                if verbose:
                    print(response_text)
                time_elapsed = timer.get_time()
                metadata = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cost": self.calculate_cost(
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                        model_meta=model_meta,
                    ),
                    "inference_time": time_elapsed,
                    "model": model,
                }
                return response_text, metadata

            except Exception as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    raise e

                ### EXPONENTIAL BACKOFF -- 1s, 2s, 4s ...
                delay = min(self.initial_wait * (2 ** (retry_count - 1)), self.max_wait)
                await asyncio.sleep(delay)
        raise Exception("Max retries exceeded")

    ########################
    ### OPENAI INFERENCE ###
    ########################
    ### ASYNCHRONOUS ONE-SHOT WITH RETRY
    async def openai_acomplete(
        self, text, system_prompt=None, model=None, verbose=True, numerical=False
    ):
        timer = Timer()

        ### SET MODEL
        model = model if model else self.model_config["default_models"]["openai"]
        if model not in self.model_config["models"]["openai"].keys():
            raise ValueError("Invalid OpenAI model name provided.")
        model_meta = self._get_model_meta(model)

        ### ALLOW FOR ASYNC RETRY ON API ERRORS
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                ### GENERATE MESSAGE THREAD
                system_template = (
                    system_prompt
                    if system_prompt
                    else self._read_prompt_template("default_prompt")
                )
                message_thread = [
                    {"role": "system", "content": system_template},
                    {"role": "user", "content": text},
                ]

                ### OPENAI ASYNC COMPLETION
                response = await self.openai_aclient.chat.completions.create(
                    model=model_meta["model_id"],
                    messages=message_thread,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                text_response = response.choices[0].message.content

                ### VALIDATE SQL ID NECESSARY
                if numerical:
                    try:
                        text_response = self._validate_float(text_response)
                    except Exception as e:
                        raise ValueError(f"Invalid float generated: {text_response}")

                ### COMPILE METADATA AND RETURN
                if verbose:
                    print(text_response)
                time_elapsed = timer.get_time()
                metadata = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "cost": self.calculate_cost(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                        model_meta=model_meta,
                    ),
                    "inference_time": time_elapsed,
                    "model": model,
                }
                return text_response, metadata

            except Exception as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    raise e

                ### EXPONENTIAL BACKOFF -- 1s, 2s, 4s ...
                delay = min(self.initial_wait * (2 ** (retry_count - 1)), self.max_wait)
                await asyncio.sleep(delay)
        raise Exception("Max retries exceeded")

    #########################
    ### BEDROCK INFERENCE ###
    #########################
    def bedrock_list_models(self):
        ### PRINT AND LIST THE AVAILABLE BEDROCK MODELS
        available_models = [
            models["modelId"]
            for models in self.bedrock_base_client.list_foundation_models()[
                "modelSummaries"
            ]
        ]
        for i in available_models:
            print(i)

    ### ASYNCHRONOUS ONE-SHOT WITH RETRY
#     async def bedrock_acomplete(
#         self, text, system_prompt=None, model=None, verbose=True, numerical=False
#     ):
#         timer = Timer()

#         ### SET MODEL
#         model = model if model else self.model_config["default_models"]["bedrock"]
#         if model not in self.model_config["models"]["bedrock"].keys():
#             raise ValueError("Invalid AWS Bedrock model name provided.")
#         model_meta = self._get_model_meta(model)

#         ### ALLOW FOR ASYNC RETRY ON API ERRORS
#         retry_count = 0
#         while retry_count < self.max_retries:
#             try:
#                 ### GENERATE MESSAGE THREAD
#                 system_template = (
#                     system_prompt
#                     if system_prompt
#                     else self._read_prompt_template("default_prompt")
#                 )
#                 model_provider = model_meta["model_id"].split(".")[0]
#                 if model_provider == "us":
#                     model_provider = model_meta["model_id"].split(".")[
#                         1
#                     ]  # edge case for anthropic bedrock models

#                 ### FORMAT PROMPT ACCORDING TO PROVIDER
#                 if model_provider == "meta":
#                     bedrock_body = self._format_meta(system_template, text, model_meta)
#                 elif model_provider == "anthropic":
#                     bedrock_body = self._format_anthropic(
#                         system_template, text, model_meta
#                     )
#                 else:
#                     raise ValueError(
#                         f"Model provider ({model_provider}) not recognized."
#                     )

#                 ### BEDROCK ASYNC COMPLETION
#                 # Start the asynchronous job
#                 s3_bucket = os.environ.get("BEDROCK_S3_BUCKET", None)
#                 s3_prefix = os.environ.get("BEDROCK_S3_PREFIX", None)
#                 if not s3_bucket or not s3_prefix:
#                     raise ValueError("Please set `BEDROCK_S3_BUCKET` and `BEDROCK_S3_PREFIX` to the .env.")

#                 response = await self.bedrock_client.start_async_invoke(
#                     modelId=model_meta["model_id"],
#                     modelInput={
#                         "contentType": "application/json",
#                         "body": json.dumps(bedrock_body)  # Ensure the body is JSON serialized
#                     },
#                     outputDataConfig={
#                         "s3OutputDataConfig": {
#                             "s3Uri": "s3://your-bucket/your-prefix/"
#                         }
#                     },
#                     clientRequestToken=f"request-{uuid.uuid4()}"  # Optional but recommended for tracking
#                 )

#                 # Get the response ID
#                 async_response_id = response["responseId"]

#                 # Poll for completion
#                 max_poll_attempts = 30
#                 poll_interval = 2  # seconds
#                 for attempt in range(max_poll_attempts):
#                     status_response = await self.bedrock_client.get_async_invoke_status(
#                         responseId=async_response_id
#                     )

#                     status = status_response["status"]
#                     if status == "COMPLETED":
#                         break
#                     elif status in ["FAILED", "EXPIRED", "STOPPED"]:
#                         error_message = status_response.get("failureReason", "Unknown error")
#                         raise ValueError(f"Async inference failed: {error_message}")

#                     await asyncio.sleep(poll_interval)
#                     poll_interval = min(poll_interval * 1.5, 10)  # Exponential backoff

#                 # Get the output
#                 output_response = await self.bedrock_client.get_async_invoke_output(
#                     responseId=async_response_id
#                 )

#                 # Process the output
#                 output_body = output_response["body"]

#                 ### PARSE THROUGH RESPONSE OUTPUT
#                 try:
#                     if model_provider == "meta":
#                         final_message, input_tokens, output_tokens = (
#                             self._handle_meta_response(output_body, verbose)
#                         )
#                     elif model_provider == "anthropic":
#                         final_message, input_tokens, output_tokens = (
#                             self._handle_anthropic_response(output_body, verbose)
#                         )
#                 except Exception as e:
#                     raise ValueError(f"Failed to parse Bedrock completion: {e}")

#                 ### VALIDATE SQL ID NECESSARY
#                 if numerical:
#                     try:
#                         final_message = self._validate_float(final_message)
#                     except Exception as e:
#                         raise ValueError(f"Invalid float response generated: {final_message}")

#                 ### COMPILE METADATA AND RETURN
#                 time_elapsed = timer.get_time()
#                 metadata = {
#                     "input_tokens": input_tokens,
#                     "output_tokens": output_tokens,
#                     "cost": self.calculate_cost(
#                         input_tokens,
#                         output_tokens,
#                         model_meta=model_meta,
#                     ),
#                     "inference_time": time_elapsed,
#                     "model": model,
#                 }
#                 return final_message, metadata

#             except Exception as e:
#                 retry_count += 1
#                 if retry_count == self.max_retries:
#                     raise e

#                 ### EXPONENTIAL BACKOFF -- 1s, 2s, 4s ...
#                 delay = min(self.initial_wait * (2 ** (retry_count - 1)), self.max_wait)
#                 await asyncio.sleep(delay)
#         raise Exception("Max retries exceeded")
    
    
    async def bedrock_acomplete(
        self, text, system_prompt=None, model=None, verbose=True, numerical=False
    ):
        timer = Timer()

        ### SET MODEL
        model = model if model else self.model_config["default_models"]["bedrock"]
        if model not in self.model_config["models"]["bedrock"].keys():
            raise ValueError("Invalid AWS Bedrock model name provided.")
        model_meta = self._get_model_meta(model)

        ### ALLOW FOR ASYNC RETRY ON API ERRORS
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                ### GENERATE MESSAGE THREAD
                system_template = (
                    system_prompt
                    if system_prompt
                    else self._read_prompt_template("default_prompt")
                )
                model_provider = model_meta["model_id"].split(".")[0]
                if model_provider == "us":
                    model_provider = model_meta["model_id"].split(".")[
                        1
                    ]  # edge case for anthropic bedrock models

                ### FORMAT PROMPT ACCORDING TO PROVIDER
                if model_provider == "meta":
                    bedrock_body = self._format_meta(system_template, text, model_meta)
                elif model_provider == "anthropic":
                    bedrock_body = self._format_anthropic(
                        system_template, text, model_meta
                    )
                else:
                    raise ValueError(
                        f"Model provider ({model_provider}) not recognized."
                    )

                ## BEDROCK ASYNC COMPLETION
                def bedrock_sync_completion(bedrock_body, model_meta):
                    response = self.bedrock_client.invoke_model(
                        body=bedrock_body,
                        modelId=model_meta["model_id"],
                        accept="application/json",
                        contentType="application/json",
                    )
                    return response

                ### EXECUTE BEDROCK INFERENCE ASYNCHRONOUSLY
                response = await asyncio.to_thread(
                    bedrock_sync_completion, bedrock_body, model_meta
                )
                
                # async def bedrock_async_completion(bedrock_body, model_meta):
                #     response = await self.bedrock_client.start_async_invoke(
                #         body=bedrock_body,
                #         modelId=model_meta["model_id"],
                #         accept="application/json",
                #         contentType="application/json",
                #     )
                #     return response
                # response = await bedrock_async_completion(bedrock_body, model_meta)
                ### !! NEED TO FIX TO PROPERLY USE THE BEDROCK ASYNC API ... UNNECESSARY FOR NOW ... !!
                
                ### PARSE THROUGH RESPONSE OUTPUT
                try:
                    if model_provider == "meta":
                        final_message, input_tokens, output_tokens = (
                            self._handle_meta_response(response, verbose)
                        )
                    elif model_provider == "anthropic":
                        final_message, input_tokens, output_tokens = (
                            self._handle_anthropic_response(response, verbose)
                        )
                except Exception as e:
                    raise ValueError(f"Failed to parse Bedrock completion: {e}")

                ### VALIDATE SQL ID NECESSARY
                if numerical:
                    try:
                        final_message = self._validate_float(final_message)
                    except Exception as e:
                        raise ValueError(f"Invalid float response generated: {final_message}")

                ### COMPILE METADATA AND RETURN
                time_elapsed = timer.get_time()
                metadata = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": self.calculate_cost(
                        input_tokens,
                        output_tokens,
                        model_meta=model_meta,
                    ),
                    "inference_time": time_elapsed,
                    "model": model,
                }
                return final_message, metadata

            except Exception as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    raise e

                ### EXPONENTIAL BACKOFF -- 1s, 2s, 4s ...
                delay = min(self.initial_wait * (2 ** (retry_count - 1)), self.max_wait)
                await asyncio.sleep(delay)
        raise Exception("Max retries exceeded")


    def _handle_meta_response(self, response, verbose=True):
        response_body = json.loads(response["body"].read())
        final_message = response_body.get("generation")
        input_tokens = response_body.get("prompt_token_count")
        output_tokens = response_body.get("generation_token_count")
        if verbose:
            print(final_message, end="")
        return final_message, input_tokens, output_tokens

    def _handle_anthropic_response(self, response, verbose=True):
        response_body = json.loads(response["body"].read())
        final_message = response_body.get("content")[0].get("text")
        input_tokens = response_body.get("usage").get("input_tokens")
        output_tokens = response_body.get("usage").get("output_tokens")
        if verbose:
            print(final_message, end="")
        return final_message, input_tokens, output_tokens

    ### LLAMA MODEL PROMPT FORMATTING
    def _format_meta(self, system_template, text, model_meta):
        prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_template}<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return json.dumps(
            {
                "prompt": prompt,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_gen_len": model_meta["max_tokens"],
            }
        )

    ### BEDROCK SONNET MODEL PROMPT FORMATTING
    def _format_anthropic(self, system_template, text, model_meta):
        return json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": model_meta["max_tokens"],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"System: {system_template}"}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text}],
                    },
                ],
            }
        )

    ### MISTRAL PROMPT BODY FORMATTING
    def _format_mistral(self, system_template, text, model_meta):
        prompt = f"<s>{system_template}\n[INST]{text}[/INST]"
        return json.dumps(
            {
                "prompt": prompt,
                "max_tokens": model_meta["max_tokens"],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
            }
        )

    #########################
    ### VECTOR EMBEDDINGS ###
    #########################
    def openai_embedding(self, text, dimensions=None):
        ### SET DIMENSIONS
        dimensions = dimensions if dimensions else self.openai_embedding_dimensions
        text = text.replace("\n", " ")
        return (
            self.openai_client.embeddings.create(
                input=[text],
                model=self.openai_embedding_model,
                dimensions=dimensions,
            )
            .data[0]
            .embedding
        )

    def bedrock_embedding(self, text, dimensions=None):
        ### SET DIMENSIONS
        dimensions = dimensions if dimensions else self.bedrock_embedding_dimensions
        text = text.replace("\n", " ")

        ### INVOKE BEDROCK EMBEDDING MODEL
        try:
            response = self.bedrock_client.invoke_model(
                body=json.dumps({"inputText": text}),
                modelId=self.bedrock_embedding_model,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            embedding = response_body.get("embedding")
            return embedding
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "AccessDeniedException":
                print(
                    f"\x1b[41m{error.response['Error']['Message']}\
                        \nTo troubeshoot this issue please refer to the following resources.\
                         \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                         \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"
                )
            else:
                raise error

    ###########
    ### BIN ###
    ###########
    def _form_message(self, content, role="user"):
        return [
            {"role": role, "content": content},
        ]

    def _read_prompt_template(self, prompt_name):
        template_path = os.path.join(self.prompt_template_location, f"{prompt_name}.txt")
        with open(template_path, "r") as file:
            prompt = file.read()
        return prompt

    def _validate_float(self, input_string):
        ### VALIDATE FLOATING POINT RESPONSE
        float_pattern = r"-?\d+\.\d+"
        matches = re.findall(float_pattern, input_string)
        if len(matches) == 0:
            raise ValueError("No floating-point number found in the input string.")
        elif len(matches) > 1:
            raise ValueError(
                "Multiple floating-point numbers found in the input string."
            )
        return float(matches[0])

    def calculate_cost(self, input_tokens, output_tokens, model_meta):
        return (input_tokens * float(model_meta["input_token_cost"])) + (
            output_tokens * float(model_meta["output_token_cost"])
        )

    ###############
    ### TESTING ###
    ###############
    async def atest_acomplete(self):
        ### LOOP THROUGH ALL MODELS AND ASYNC COMPLETE
        for model in self.all_models:
            try:
                print(f"\n{model}\n")
                response, metadata = await self.acomplete(
                    "Say hello", model=model, verbose=True
                )
                print("\n")
                print(json.dumps(metadata))
                print("\n------\n")
            except Exception as e:
                print(f"An ERROR occured: {e}")
                continue

    def test_acomplete(self):
        asyncio.run(self.atest_acomplete())




#######################
### UNIVERSAL TIMER ###
#######################
class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_time(self):
        end = time.time()
        return round(end - self.start, 2)