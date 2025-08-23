#!/usr/bin/env python
# coding: utf-8

# # Using Function Calling For Structure

# ## Simple Example Of Address Extraction

# In[1]:


text = \
"""
John Doe lives at 123 Elm Street, Springfield. Next to him is Jane Smith, residing at 456 Oak Avenue, Lakeview. Not far away, we find Dr. Emily Ryan at 789 Pine Road, Westwood. Meanwhile, in a different part of town, Mr. Alan Turing can be found at 101 Binary Blvd, Computerville. Nearby, Ms. Olivia Newton stays at 202 Music Lane, Harmony. Also, Prof. Charles Xavier is located at 505 Mutant Circle, X-Town.
"""
print (text)


# In[2]:


raven_prompt = \
f'''
Function:
def address_name_pairs(names : list[str], addresses : list[str]):
"""
Give names and associated addresses.
"""

{text}<human_end>
'''


# In[3]:


from utils import query_raven

def address_name_pairs(names : list[str], addresses : list[str]):
  """
  Give names and associated addresses.
  """
  for name, addr in zip(names, addresses):
    print (name, ": ", addr)

result = query_raven(raven_prompt)
eval(result)


# ## Alternative Way of Doing Extraction

# In[ ]:


unbalanced_text = \
"""
Dr. Susan Hill has a practice at 120 Green Road, Evergreen City, and also consults at 450 Riverdale Drive, Brookside. Mark Twain, the renowned author, once lived at 300 Maple Street, Springfield, but now resides at 200 Writers Block, Literaryville. The famous artist, Emily Carter, showcases her work at 789 Artisan Alley, Paintown, and has a studio at 101 Palette Place, Creativeland. Meanwhile, the tech innovator, John Tech, has his main office at 555 Silicon Street, Techville, and a secondary office at 777 Data Drive, Computown, but he lives at 123 Digital Domain, Innovatown.
"""
print (unbalanced_text)


# In[ ]:


raven_prompt = \
f'''

@dataclass
class Record:
    name : str
    addresses : List[str]

Function:
def insert_into_database(names : List[Record]):
"""
Inserts the records into the database. 
"""

{unbalanced_text}<human_end>

'''

result = query_raven(raven_prompt)
print (result)


# ## Generating Valid JSONs

# ```
# {
#   "city_name" : "London"
#   "location" : {
#       "country" : "United Kingdom",
#       "continent" : {
#           "simple_name" : "Europe",
#           "other_name" : "Afro-Eur-Asia"
#       }
#   }
# }
# ```
# 

# In[ ]:


def city_info(city_name : str, location : dict):
  """
  Gets the city info
  """
  return locals()
def construct_location_dict(country : str, continent : dict):
  """
  Provides the location dictionary
  """
  return locals()
def construct_continent_dict(simple_name : str, other_name : str):
  """
  Provides the continent dict
  """
  return locals()


# In[ ]:


print (city_info("London", {}))


# In[ ]:


raven_prompt = \
'''
Function:
def city_info(city_name : str, location : dict):
"""
Gets the city info
"""

Function:
def construct_location_dict(country : str, continent : dict):
"""
Provides the location dictionary
"""

def construct_continent_dict(simple_name : str, other_name : str):
"""
Provides the continent dict
"""

User Query: {question}<human_end>
'''


# In[ ]:


question = "I want the city info for London, "\
"which is in the United Kingdom, which is in Europe or Afro-Eur-Asia."

output = query_raven(raven_prompt.format(question = question))
json0 = eval(output)
print (json0)


# In[ ]:


import json
json.dumps(json0)


# ### Try These yourself!

# In[ ]:


question = "I need details for the city of Tokyo, "\
"situated in Japan, a part of the Asian continent, "\
"which is sometimes referred to as Eurasia."

output = query_raven(raven_prompt.format(question = question))
json1 = eval(output)
print (json1)


# In[ ]:


import json
json.dumps(json0)


# In[ ]:




