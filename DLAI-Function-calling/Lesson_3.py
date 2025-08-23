#!/usr/bin/env python
# coding: utf-8

# ### How Do We Interface With External Resources?

# In[1]:


import requests

url = "https://v2.jokeapi.dev/joke/Any?safe-mode&type=twopart"

response = requests.get(url)

print(response.json()["setup"])
print(response.json()["delivery"])


# In[2]:


response.json()


# In[3]:


import requests
def give_joke(category : str):
    """
    Joke categories. Supports: Any, Misc, Programming, Pun, Spooky, Christmas.
    """

    url = f"https://v2.jokeapi.dev/joke/{category}?safe-mode&type=twopart"
    response = requests.get(url)
    print(response.json()["setup"])
    print(response.json()["delivery"])


# In[4]:


USER_QUERY = "Hey! Can you get me a joke for this december?"


# In[6]:


import inspect
# Specify the LLM Endpoint
# Now, let's prompt Raven!
API_URL = "http://nexusraven.nexusflow.ai"
headers = {
        "Content-Type": "application/json"
}

def raven_post(payload):
	"""
	Sends a payload to a TGI endpoint.
	"""
	import requests
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def query_raven(prompt):
	"""
	This function sends a request to the TGI endpoint to get Raven's function call.
	This will not generate Raven's justification and reasoning for the call, to save on latency.
	"""
	import requests
	output = raven_post({
		"inputs": prompt,
		"parameters" : {"temperature" : 0.001, "stop" : ["<bot_end>"], "do_sample" : False, "max_new_tokens" : 2048, "return_full_text" : False}})
	call = output[0]["generated_text"].replace("Call:", "").strip()
	return call

def clean_docstring(docstring):
    if docstring is not None:
        # Remove leading and trailing whitespace
        docstring = docstring.strip()
    return docstring

def construct_prompt(raven_msg, functions):
    full_prompt = ""
    for function in functions:
        signature = inspect.signature(function)
        docstring = function.__doc__
        prompt = f'''Function:\n{function.__name__}{signature}\n"""{clean_docstring(docstring)}"""'''
        full_prompt += prompt + "\n\n"
    full_prompt += f'''User Query: {raven_msg}<human_end>'''
    return full_prompt


# In[ ]:


from utils import query_raven

raven_functions = \
f'''
def give_joke(category : str):
    """
    Joke categories. Supports: Any, Misc, Programming, Dark, Pun, Spooky, Christmas.
    """

User Query: {USER_QUERY}<human_end>
'''
call = query_raven(raven_functions)


# In[ ]:


exec(call)


# #### Writing A Tool That Uses OpenAPI APIs

# In[ ]:


#!wget https://raw.githubusercontent.com/open-meteo/open-meteo/main/openapi.yml


# In[ ]:


import yaml
import json

# Read the content of the file
with open('openapi.yml', 'r') as file:
    file_content = file.read()
file_content = file_content.replace("int\n", "number\n")
file_content = file_content.replace("float\n", "number\n")
data = yaml.safe_load(file_content)

data["servers"] = [{"url":"https://api.open-meteo.com"}]

with open('openapi.json', 'w') as file:
    json_content = json.dump(data, file)


# In[ ]:


get_ipython().system('openapi-python-generator openapi.json ./api_specification_main/')


# In[ ]:


from api_specification_main.services.WeatherForecastAPIs_service\
    import get_v1forecast


# In[ ]:


user_query = "Hey how is the current weather and windspeed in New York?"


# In[ ]:


import inspect
signature = inspect.signature(get_v1forecast)
docstring = \
'''
Requires the latitude and longitude.
Set current_weather to True to get the weather.
Set hourly or daily based on preference.
'''

raven_prompt = \
f'''
Function:
{get_v1forecast.__name__}{signature}
"""{docstring}"""

User Query: {user_query}<human_end>'''

print (raven_prompt)


# In[ ]:


from utils import query_raven
call = query_raven(raven_prompt)
print (call)


# In[ ]:


eval(call)


# In[ ]:




