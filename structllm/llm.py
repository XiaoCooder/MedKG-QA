import openai 
import time
import requests
from openai import OpenAI
import os


class gpt(object):
    def __init__(self, args):

        os.environ["OPENAI_BASE_URL"] = args.openai_url
        os.environ["OPENAI_API_KEY"] = args.key

        self.model = args.model
        OpenAI.api_key = args.key
        self.url = args.openai_url

        self.client = OpenAI(base_url=self.url, api_key=args.key)

        """
        else:
          pass
          OpenAI.api_key = args.embedding_key
          self.url = args.embedding_url
          self.client = OpenAI(base_url=args.embedding_url, api_key=args.embedding_key)
          self.model = args.embedding_model
        """

    def get_response(self, prompt, flag=0, num=1):

        if self.model=="qwen-max" or type(prompt)==str:
            start_time = time.time()
            while True:
                try:
                    if time.time() - start_time > 300:  # 300 seconds = 5 minutes
                        raise TimeoutError("Code execution exceeded 5 minutes")

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=prompt,
                        temperature=0.5, #1
                        max_tokens=512,
                        top_p=0.5,
                        #frequency_penalty=1.05,
                        presence_penalty=0,
                        n=1
                    )
                    
                    return response.choices[0].message.content
                
                except requests.exceptions.RequestException as e:
                    # 如果发生网络异常，等待10秒后重试
                    print(f"Network error: {e}")
                    print("Retrying in 10 seconds...")
                    time.sleep(10)

                except openai.APIError as e:
                    print('OpenAI.APIError\nRetrying...')
                    print(e)
                    time.sleep(20)

                except openai.APIConnectionError as e:
                    print('OpenAI.APIConnectionError\n{e}\nRetrying...')
                    time.sleep(20)

                except openai.RateLimitError as e:
                    err_mes = str(e)
                    if "You exceeded your current quota" in err_mes:
                        print("You exceeded your current quota")
                    print('OpenAI.error.RateLimitError\nRetrying...')
                    time.sleep(30)

                except openai.APITimeoutError:
                    print('OpenAI.APITimeoutError\nRetrying...')
                    time.sleep(20)

                except TimeoutError as e:
                    # Handle the custom TimeoutError exception
                    print(f"Code execution exceeded 5 minutes: {e}")
                    # Optionally, you can re-raise the exception to terminate the script
                    raise e 
        else:
            start_time = time.time()
            while True:
                try:
                    if time.time() - start_time > 300:  # 300 seconds = 5 minutes
                        raise TimeoutError("Code execution exceeded 5 minutes")

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=prompt,
                        temperature=0.7, #1
                        max_tokens=2048,
                        top_p=0.6,
                        #frequency_penalty=1.05,
                        presence_penalty=0,
                        stream=False,
                    )
                    #print(response)
                    return [response]

                except requests.exceptions.RequestException as e:
                    # 如果发生网络异常，等待10秒后重试
                    print(f"Network error: {e}")
                    print("Retrying in 10 seconds...")
                    time.sleep(10)

                except openai.APIError as e:
                    print('OpenAI.APIError\nRetrying...')
                    print(e)
                    time.sleep(20)

                except openai.APIConnectionError as e:
                    print('OpenAI.APIConnectionError\n{e}\nRetrying...')
                    time.sleep(20)

                except openai.RateLimitError as e:
                    err_mes = str(e)
                    if "You exceeded your current quota" in err_mes:
                        print("You exceeded your current quota")
                    print('OpenAI.error.RateLimitError\nRetrying...')
                    time.sleep(30)

                except openai.APITimeoutError:
                    print('OpenAI.APITimeoutError\nRetrying...')
                    time.sleep(20)

                except TimeoutError as e:
                    # Handle the custom TimeoutError exception
                    print(f"Code execution exceeded 5 minutes: {e}")
                    # Optionally, you can re-raise the exception to terminate the script
                    raise e
                