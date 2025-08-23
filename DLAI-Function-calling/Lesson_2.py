#!/usr/bin/env python
# coding: utf-8

# # Function Calling

# ### Housekeeping

# In[1]:


def afunction(arg1:int = 0, arg2:str = "hello", **kwargs)->int:
    ''' this is a function definition
        arg1 (int): an exemplary yet modest argument
        arg2 (str): another nice argument
        **kwargs : the rest of the rabble 

        returns arg1 incremented by one
    '''
    return(arg + 1)


# In[2]:


print(afunction.__name__)
print(afunction.__doc__)


# In[3]:


import inspect
print(inspect.signature(afunction))


# ##### Building User Query
# > Note, the video shows how to access utils.py in Jupyter Notebook V7.   
# > You may be in Jupyter Notebook V6. To access it in this version, on the menu, choose File->Open. 

# In[4]:


import inspect
def build_raven_prompt(function_list, user_query):
    raven_prompt = ""
    for function in function_list:
        signature = inspect.signature(function)
        docstring = function.__doc__
        prompt = \
f'''
Function:
def {function.__name__}{signature}
    """
    {docstring.strip()}
    """
    
'''
        raven_prompt += prompt
        
    raven_prompt += f"User Query: {user_query}<human_end>"
    return raven_prompt


# In[5]:


print( build_raven_prompt([afunction], "a query"))


# ### Concrete Example For Parallel Calls

# In[6]:


from utils import draw_clown_face

raven_msg = "Hey, can you build me two clowns." \
"The first clown should be red faced, with a blue nose" \
"and a mouth from 0 to 180 degrees. The mouth should be black." \
"The second clown should have a blue face and a green nose" \
"and a red mouth that's 180 to 360 degrees."


# #### Building Raven Prompt

# In[7]:


raven_prompt = build_raven_prompt([draw_clown_face], raven_msg)

print (raven_prompt)


# In[ ]:


from utils import query_raven
raven_call = query_raven(raven_prompt)
print (raven_call)
exec(raven_call)


# ### Using Multiple Functions!

# #### Building The Prompt

# In[9]:


from utils import draw_clown_face, draw_tie
raven_msg = "Hey draw a tie?"
raven_prompt = build_raven_prompt\
    ([draw_clown_face, draw_tie], raven_msg)


# In[10]:


print(raven_prompt)


# #### Getting The Call

# In[ ]:


raven_call = query_raven(raven_prompt)
print (raven_call)


# In[ ]:


exec(raven_call)


# ### Multiple Parallel Function Calling

# #### Build The Prompt

# In[ ]:


raven_msg = "Draw a clown and a tie?"


# In[ ]:


raven_prompt = build_raven_prompt([draw_tie, draw_clown_face], raven_msg)
raven_call = query_raven(raven_prompt)


# In[ ]:


print (raven_call)


# In[ ]:


exec(raven_call)


# ### What is the significance of the docstrings?

# In[ ]:


raven_msg = "Draw me a sad one with green head"
raven_prompt = build_raven_prompt([draw_clown_face], raven_msg)
raven_call = query_raven(raven_prompt)
print (raven_call)
exec(raven_call)


# #### Fixing The Function Docstring

# In[ ]:


print (raven_prompt)


# In[ ]:


raven_prompt_targeted = \
'''
Function:
def draw_clown_face(face_color='yellow', eye_color='black', nose_color='red', eye_size=0.05, mouth_size=(0.3, 0.1), mouth_color='black', eye_offset=(0.15, 0.15), mouth_theta=(200, 340))
    """
    Draws a customizable, simplified clown face using matplotlib.

    Parameters:
    - face_color (str): Color of the clown's face. Default is 'yellow'.
    - eye_color (str): Color of the clown's eyes. Default is 'black'.
    - nose_color (str): Color of the clown's nose. Default is 'red'.
    - eye_size (float): Radius of the clown's eyes. Default is 0.05.
    - mouth_size (tuple): Width and height of the clown's mouth arc. Default is (0.3, 0.1).
    - eye_offset (tuple): Horizontal and vertical offset for the eyes from the center. Default is (0.15, 0.15).
    - mouth_theta (tuple): Controls the emotions of the clown. Starting and ending angles (in degrees) of the mouth arc. Default is (200, 340).

    This function creates a plot displaying a simplified clown face, where essential facial features' size, position, and color can be customized. 

    Example usage:
    draw_clown_face(face_color='lightblue', eye_color='green', nose_color='orange', 
                    eye_size=0.07, mouth_size=(0.4, 0.25), 
                    eye_offset=(0.2, 0.2), mouth_theta=(0, 180))

    # This will draw a simplified clown face with a light blue face, green eyes, an orange nose, and a smiling mouth.
    """
    
User Query: Draw me a sad one with green head<human_end>
'''


# In[ ]:


raven_call = query_raven(raven_prompt_targeted)
print (raven_call)
exec(raven_call)


# ### Concrete Example For Nested APIs

# In[ ]:


raven_msg_nested = "Hey draw me a clown with "\
                    "a red face, blue eyes, green nose, "\
                    "and a black mouth open from 0 to 360 "\
                    "degrees?"


# In[ ]:


from utils import draw_head, draw_eyes, \
    draw_nose, draw_mouth, \
    draw_clown_face_parts
raven_function_nested = build_raven_prompt([draw_head, \
                                            draw_eyes, \
                                            draw_nose, \
                                            draw_mouth, \
                                            draw_clown_face_parts],\
                                            raven_msg_nested)
raven_call = query_raven(raven_function_nested)


# In[ ]:


print (raven_call)


# In[ ]:


exec(raven_call)


# In[ ]:





# In[ ]:





# In[ ]:


import inspect
def raven_post(payload):
    """
    Sends a payload to a TGI endpoint.
    """
    # Now, let's prompt Raven!
    API_URL = "http://nexusraven.nexusflow.ai"
    headers = {
            "Content-Type": "application/json"
    }
    import requests
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def call_functioncalling_llm(prompt, api_to_call):
    """
    This function sends a request to the TGI endpoint to get Raven's function call.
    This will not generate Raven's justification and reasoning for the call, to save on latency.
    """
    signature = inspect.signature(api_to_call)
    docstring = api_to_call.__doc__
    prompt = f'''Function:\n{api_to_call.__name__}{signature}\n"""{clean_docstring(docstring)}"""\n\n\nUser Query:{prompt}<human_end>'''
    import requests
    output = raven_post({
        "inputs": prompt,
        "parameters" : {"temperature" : 0.001, "stop" : ["<bot_end>"], "do_sample" : False, "max_new_tokens" : 2048, "return_full_text": False}})
    call = output[0]["generated_text"].replace("Call:", "").strip()
    return call

def query_raven(prompt):
	"""
	This function sends a request to the TGI endpoint to get Raven's function call.
	This will not generate Raven's justification and reasoning for the call, to save on latency.
	"""
	import requests
	output = raven_post({
		"inputs": prompt,
		"parameters" : {"temperature" : 0.001, "stop" : ["<bot_end>"], "return_full_text" : False, "do_sample" : False, "max_new_tokens" : 2048}})
	call = output[0]["generated_text"].replace("Call:", "").strip()
	return call

def clean_docstring(docstring):
    if docstring is not None:
        # Remove leading and trailing whitespace
        docstring = docstring.strip()
    return docstring

def build_raven_prompt(function_list, user_query):
    import inspect
    raven_prompt = ""
    for function in function_list:
        signature = inspect.signature(function)
        docstring = function.__doc__
        prompt = \
f'''
Function:
def {function.__name__}{signature}
    """
    {clean_docstring(docstring)}
    """
    
'''
        raven_prompt += prompt
        
    raven_prompt += f"User Query: {user_query}<human_end>"
    return raven_prompt

