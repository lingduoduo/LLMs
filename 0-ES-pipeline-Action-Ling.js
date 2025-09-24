{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "2fd9c253",
   "metadata": {},
   "source": [
    "### Capstion Data - English"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2428650c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "data = pd.read_csv(\"capstone_full_data2.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "901f219c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 23165 entries, 0 to 23164\n",
      "Data columns (total 9 columns):\n",
      " #   Column      Non-Null Count  Dtype \n",
      "---  ------      --------------  ----- \n",
      " 0   category    23165 non-null  object\n",
      " 1   int_id      23165 non-null  int64 \n",
      " 2   object_id   23165 non-null  object\n",
      " 3   context     185 non-null    object\n",
      " 4   caption     23165 non-null  object\n",
      " 5   subtitle    23161 non-null  object\n",
      " 6   location    23165 non-null  object\n",
      " 7   legal_name  21371 non-null  object\n",
      " 8   keywords    150 non-null    object\n",
      "dtypes: int64(1), object(8)\n",
      "memory usage: 1.6+ MB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b32eab62",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "category                                                actions\n",
       "int_id                                                        4\n",
       "object_id                      0429ea12a5974851ab47620c9d7205c9\n",
       "context                                                  global\n",
       "caption                            View Criteria Control Center\n",
       "subtitle      Setup and manage the search queries that give ...\n",
       "location                         135 West 18th Street, NY 10010\n",
       "legal_name                                                  NaN\n",
       "keywords      criteria search query queries flex structures ...\n",
       "Name: 0, dtype: object"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.iloc[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "315a93fa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "category\n",
       "people       22976\n",
       "actions        185\n",
       "mini-apps        4\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['category'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "8bd3b308",
   "metadata": {},
   "outputs": [],
   "source": [
    "action_data = data[data.category=='actions']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "64a4be62",
   "metadata": {},
   "outputs": [],
   "source": [
    "action_data = action_data.drop(columns=['legal_name'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d255cea1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# row_security = [\"H1\" if i%2==0 else \"H2\" for i in range(len(action_data))]\n",
    "# row_security = [\"H\" for i in range(len(action_data))]\n",
    "row_security = [[\"H\", \"H1\"] if i%100==0 else \"H\" for i in range(len(action_data))]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0d35cffa",
   "metadata": {},
   "outputs": [],
   "source": [
    "action_data = action_data.assign(row_security=row_security)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2beeb7ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>category</th>\n",
       "      <th>int_id</th>\n",
       "      <th>object_id</th>\n",
       "      <th>context</th>\n",
       "      <th>caption</th>\n",
       "      <th>subtitle</th>\n",
       "      <th>location</th>\n",
       "      <th>keywords</th>\n",
       "      <th>row_security</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>actions</td>\n",
       "      <td>4</td>\n",
       "      <td>0429ea12a5974851ab47620c9d7205c9</td>\n",
       "      <td>global</td>\n",
       "      <td>View Criteria Control Center</td>\n",
       "      <td>Setup and manage the search queries that give ...</td>\n",
       "      <td>135 West 18th Street, NY 10010</td>\n",
       "      <td>criteria search query queries flex structures ...</td>\n",
       "      <td>[H, H1]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>actions</td>\n",
       "      <td>103</td>\n",
       "      <td>9010674ab9504e8a8d892f01286f170f</td>\n",
       "      <td>global</td>\n",
       "      <td>View My Policies</td>\n",
       "      <td>View My Policies</td>\n",
       "      <td>135 West 18th Street, NY 10010</td>\n",
       "      <td>NaN</td>\n",
       "      <td>H</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>actions</td>\n",
       "      <td>37</td>\n",
       "      <td>40a2fc1bb38c42889709f0407a7175b1</td>\n",
       "      <td>global</td>\n",
       "      <td>Manage Fluid Field Definitions</td>\n",
       "      <td>Create, view or modify fluid field definitions.</td>\n",
       "      <td>135 West 18th Street, NY 10010</td>\n",
       "      <td>fluid fields custom field</td>\n",
       "      <td>H</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>actions</td>\n",
       "      <td>31</td>\n",
       "      <td>3979804c3c9d4efd8dcda48fc2fb957d</td>\n",
       "      <td>global</td>\n",
       "      <td>Edit Birth Details</td>\n",
       "      <td>Edit your birth details</td>\n",
       "      <td>135 West 18th Street, NY 10010</td>\n",
       "      <td>personal detail nationality birth details</td>\n",
       "      <td>H</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>actions</td>\n",
       "      <td>67</td>\n",
       "      <td>684b1c4fce704043ab813fec1b100573</td>\n",
       "      <td>global</td>\n",
       "      <td>Associate data import</td>\n",
       "      <td>Associate data import</td>\n",
       "      <td>135 West 18th Street, NY 10010</td>\n",
       "      <td>Import associate data Associate data import As...</td>\n",
       "      <td>H</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>180</th>\n",
       "      <td>actions</td>\n",
       "      <td>157</td>\n",
       "      <td>de1c40a9c57141babef3c219285d926f</td>\n",
       "      <td>US</td>\n",
       "      <td>View Your Tax Statements</td>\n",
       "      <td>Access your tax statements.</td>\n",
       "      <td>135 West 18th Street, NY 10010</td>\n",
       "      <td>W2 tax form</td>\n",
       "      <td>H</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>181</th>\n",
       "      <td>actions</td>\n",
       "      <td>147</td>\n",
       "      <td>cc19213694c74782a79bea93eb554620</td>\n",
       "      <td>global</td>\n",
       "      <td>Manage Consent for ADP Benchmarks</td>\n",
       "      <td>Manage Consent for ADP Benchmarks</td>\n",
       "      <td>135 West 18th Street, NY 10010</td>\n",
       "      <td>NaN</td>\n",
       "      <td>H</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>182</th>\n",
       "      <td>actions</td>\n",
       "      <td>141</td>\n",
       "      <td>c05481684c1140baa97e8d5c2c798bba</td>\n",
       "      <td>global</td>\n",
       "      <td>Registration Dashboard</td>\n",
       "      <td>Dashboard to track associates registration</td>\n",
       "      <td>135 West 18th Street, NY 10010</td>\n",
       "      <td>send registration registration register</td>\n",
       "      <td>H</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23163</th>\n",
       "      <td>actions</td>\n",
       "      <td>100001</td>\n",
       "      <td>1234567891</td>\n",
       "      <td>00000</td>\n",
       "      <td>Company Links and FAQs</td>\n",
       "      <td>Links for various company perks</td>\n",
       "      <td>135 West 18th Street, NY 10010</td>\n",
       "      <td>NaN</td>\n",
       "      <td>H</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23164</th>\n",
       "      <td>actions</td>\n",
       "      <td>100002</td>\n",
       "      <td>1001001001</td>\n",
       "      <td>00000</td>\n",
       "      <td>Company Values</td>\n",
       "      <td>Comany core values</td>\n",
       "      <td>135 West 18th Street, NY 10010</td>\n",
       "      <td></td>\n",
       "      <td>H</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>185 rows × 9 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      category  int_id                         object_id context  \\\n",
       "0      actions       4  0429ea12a5974851ab47620c9d7205c9  global   \n",
       "1      actions     103  9010674ab9504e8a8d892f01286f170f  global   \n",
       "2      actions      37  40a2fc1bb38c42889709f0407a7175b1  global   \n",
       "3      actions      31  3979804c3c9d4efd8dcda48fc2fb957d  global   \n",
       "4      actions      67  684b1c4fce704043ab813fec1b100573  global   \n",
       "...        ...     ...                               ...     ...   \n",
       "180    actions     157  de1c40a9c57141babef3c219285d926f      US   \n",
       "181    actions     147  cc19213694c74782a79bea93eb554620  global   \n",
       "182    actions     141  c05481684c1140baa97e8d5c2c798bba  global   \n",
       "23163  actions  100001                        1234567891   00000   \n",
       "23164  actions  100002                        1001001001   00000   \n",
       "\n",
       "                                 caption  \\\n",
       "0           View Criteria Control Center   \n",
       "1                       View My Policies   \n",
       "2         Manage Fluid Field Definitions   \n",
       "3                     Edit Birth Details   \n",
       "4                  Associate data import   \n",
       "...                                  ...   \n",
       "180             View Your Tax Statements   \n",
       "181    Manage Consent for ADP Benchmarks   \n",
       "182               Registration Dashboard   \n",
       "23163             Company Links and FAQs   \n",
       "23164                     Company Values   \n",
       "\n",
       "                                                subtitle  \\\n",
       "0      Setup and manage the search queries that give ...   \n",
       "1                                       View My Policies   \n",
       "2        Create, view or modify fluid field definitions.   \n",
       "3                                Edit your birth details   \n",
       "4                                  Associate data import   \n",
       "...                                                  ...   \n",
       "180                          Access your tax statements.   \n",
       "181                    Manage Consent for ADP Benchmarks   \n",
       "182           Dashboard to track associates registration   \n",
       "23163                    Links for various company perks   \n",
       "23164                                 Comany core values   \n",
       "\n",
       "                             location  \\\n",
       "0      135 West 18th Street, NY 10010   \n",
       "1      135 West 18th Street, NY 10010   \n",
       "2      135 West 18th Street, NY 10010   \n",
       "3      135 West 18th Street, NY 10010   \n",
       "4      135 West 18th Street, NY 10010   \n",
       "...                               ...   \n",
       "180    135 West 18th Street, NY 10010   \n",
       "181    135 West 18th Street, NY 10010   \n",
       "182    135 West 18th Street, NY 10010   \n",
       "23163  135 West 18th Street, NY 10010   \n",
       "23164  135 West 18th Street, NY 10010   \n",
       "\n",
       "                                                keywords row_security  \n",
       "0      criteria search query queries flex structures ...      [H, H1]  \n",
       "1                                                    NaN            H  \n",
       "2                              fluid fields custom field            H  \n",
       "3              personal detail nationality birth details            H  \n",
       "4      Import associate data Associate data import As...            H  \n",
       "...                                                  ...          ...  \n",
       "180                                          W2 tax form            H  \n",
       "181                                                  NaN            H  \n",
       "182              send registration registration register            H  \n",
       "23163                                                NaN            H  \n",
       "23164                                                               H  \n",
       "\n",
       "[185 rows x 9 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "action_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "5700f995",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 185 entries, 0 to 23164\n",
      "Data columns (total 9 columns):\n",
      " #   Column        Non-Null Count  Dtype \n",
      "---  ------        --------------  ----- \n",
      " 0   category      185 non-null    object\n",
      " 1   int_id        185 non-null    int64 \n",
      " 2   object_id     185 non-null    object\n",
      " 3   context       185 non-null    object\n",
      " 4   caption       185 non-null    object\n",
      " 5   subtitle      185 non-null    object\n",
      " 6   location      185 non-null    object\n",
      " 7   keywords      150 non-null    object\n",
      " 8   row_security  185 non-null    object\n",
      "dtypes: int64(1), object(8)\n",
      "memory usage: 14.5+ KB\n"
     ]
    }
   ],
   "source": [
    "action_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1b4a2d2d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "caption\n",
       "Request Time Off                     4\n",
       "Edit My Identification               2\n",
       "View My Timecard                     2\n",
       "View My Tax Withholding              2\n",
       "Clock In/Out                         2\n",
       "                                    ..\n",
       "View Your Pay                        1\n",
       "View Your Tax Statements             1\n",
       "Manage Consent for ADP Benchmarks    1\n",
       "Registration Dashboard               1\n",
       "Company Values                       1\n",
       "Name: count, Length: 174, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "action_data['caption'].value_counts().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "80cbd940",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "subtitle\n",
       "Request Time Off                                        4\n",
       "To enroll Wisely pay card                               2\n",
       "Add or edit your Identification Documents               2\n",
       "Manage special accommodations                           2\n",
       "View My Pay                                             2\n",
       "                                                       ..\n",
       "Send a registration email to one or many associates.    1\n",
       "Access your pay details.                                1\n",
       "Access your tax statements.                             1\n",
       "Manage Consent for ADP Benchmarks                       1\n",
       "Comany core values                                      1\n",
       "Name: count, Length: 178, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "action_data['subtitle'].value_counts().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "1528c9a0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "keywords\n",
       "criteria search query queries flex structures manage view setup implementation                                                                                                                                                                                  2\n",
       "Identification Identity                                                                                                                                                                                                                                         2\n",
       "wisely                                                                                                                                                                                                                                                          2\n",
       "fluid fields custom field                                                                                                                                                                                                                                       2\n",
       "time off time request vacation sick personal day off ooo out of office away                                                                                                                                                                                     2\n",
       "                                                                                                                                                                                                                                                               ..\n",
       "add benefits administrator set up benefits administrator set up benefits plan providers set up benefits plans create benefits profiles benefits inbound data settings benefits outbound data settings benefits notification settings benefits implementation    1\n",
       "Implement Implementation Toolkit Implementation Dashboard                                                                                                                                                                                                       1\n",
       "Termination Settings                                                                                                                                                                                                                                            1\n",
       "contact email                                                                                                                                                                                                                                                   1\n",
       "                                                                                                                                                                                                                                                                1\n",
       "Name: count, Length: 139, dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "action_data['keywords'].value_counts().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "efcec336",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location\n",
       "135 West 18th Street, NY 10010    185\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "action_data['location'].value_counts().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "915ebb5c",
   "metadata": {},
   "source": [
    "### Sentence Transformer Embeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "94840fd8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# from sentence_transformers import SentenceTransformer\n",
    "\n",
    "# # Do this off the network\n",
    "# # Make sure to change the \"save\" path to somewhere that works on your machine\n",
    "# # https://bitbucket.es.ad.adp.com/projects/LIFML/repos/ml-sentence-transformers-base-container/browse/init.py\n",
    "# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')\n",
    "# model.save('/Users/huanglin/Bitbucket/multilingual-raw-data/paraphrase-multilingual-MiniLM-L12-v2')\n",
    "# model = SentenceTransformer('/Users/huanglin/Desktop/notebooks/xinpeng-multilingual-raw-data/paraphrase-multilingual-MiniLM-L12-v2')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "480ff2d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# After that you should be able to just use this code to load the model even on the internal network\n",
    "from sentence_transformers import SentenceTransformer\n",
    "# Load model\n",
    "model = SentenceTransformer('/Users/huanglin/Desktop/notebooks/xinpeng-multilingual-raw-data/paraphrase-multilingual-MiniLM-L12-v2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "55e2bfa8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[ 1.0000,  0.1547, -0.0850],\n",
      "        [ 0.1547,  1.0000,  0.1944],\n",
      "        [-0.0850,  0.1944,  1.0000]])\n"
     ]
    }
   ],
   "source": [
    "# Convert text to text embeddings\n",
    "sentence = ['Request Time Off', 'View My Policies', 'Manage Fluid Field Definitions']\n",
    "model.max_seq_length = 384\n",
    "vector = model.encode(sentence)\n",
    "vector.shape\n",
    "\n",
    "similarities = model.similarity(vector, vector)\n",
    "print(similarities)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "08e95831",
   "metadata": {},
   "outputs": [],
   "source": [
    "# from transformers import AutoTokenizer, AutoModel\n",
    "# import torch\n",
    "\n",
    "# # Mean Pooling - Take attention mask into account for correct averaging\n",
    "# def mean_pooling(model_output, attention_mask):\n",
    "#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings\n",
    "#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()\n",
    "#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)\n",
    "# Sentences we want sentence embeddings for\n",
    "# sentence = ['Request Time Off', 'View My Policies', 'Manage Fluid Field Definitions']\n",
    "\n",
    "# # Load model from HuggingFace Hub\n",
    "# tokenizer = AutoTokenizer.from_pretrained('paraphrase-multilingual-MiniLM-L12-v2')\n",
    "# transformer_model = AutoModel.from_pretrained('paraphrase-multilingual-MiniLM-L12-v2')\n",
    "\n",
    "# # Tokenize sentences\n",
    "# encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')\n",
    "\n",
    "# # Compute token embeddings\n",
    "# with torch.no_grad():\n",
    "#     model_output = transformer_model(**encoded_input)\n",
    "\n",
    "# # Perform pooling. In this case, max pooling.\n",
    "# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])\n",
    "\n",
    "# print(\"Sentence embeddings:\")\n",
    "# print(sentence_embeddings.shape)\n",
    "# similarities = model.similarity(sentence_embeddings, sentence_embeddings)\n",
    "# print(similarities)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "3a13ccc8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Batches: 100%|██████████| 6/6 [00:00<00:00,  9.07it/s]\n"
     ]
    }
   ],
   "source": [
    "caption = list(set(action_data['caption']))\n",
    "embeddings = model.encode(caption, show_progress_bar=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "09c259f4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.\n"
     ]
    }
   ],
   "source": [
    "import umap.umap_ as umap\n",
    " \n",
    "# We instantiate our UMAP model\n",
    "umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')\n",
    " \n",
    "# We fit and transform our embeddings to reduce them\n",
    "reduced_embeddings = umap_model.fit_transform(embeddings)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "18994f99",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(174, 5)"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reduced_embeddings.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "47c345cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(174,)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from hdbscan import HDBSCAN\n",
    " \n",
    "# We instantiate our HDBSCAN model\n",
    "hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom')\n",
    " \n",
    "# We fit our model and extract the cluster labels\n",
    "hdbscan_model.fit(reduced_embeddings)\n",
    "labels = hdbscan_model.labels_\n",
    "labels.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "dc024973",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    " \n",
    "# Reduce 384-dimensional embeddings to 2 dimensions for easier visualization\n",
    "reduced_embeddings = umap.UMAP(n_neighbors=15, n_components=2, \n",
    "min_dist=0.0, metric='cosine').fit_transform(embeddings)\n",
    "df = pd.DataFrame(np.hstack([reduced_embeddings, labels.reshape(-1, 1)]),\n",
    "     columns=[\"x\", \"y\", \"cluster\"]).sort_values(\"cluster\")\n",
    " \n",
    "# Visualize clusters\n",
    "df.cluster = df.cluster.astype(int).astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "b07dcc65",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "hovertemplate": "cluster=-1<br>x=%{x}<br>y=%{y}<extra></extra>",
         "legendgroup": "-1",
         "marker": {
          "color": "#636efa",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "-1",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          3.5206127166748047,
          3.5523903369903564,
          2.0176193714141846,
          1.9173798561096191,
          2.037598133087158,
          3.2995381355285645,
          1.825797200202942,
          1.9526479244232178
         ],
         "xaxis": "x",
         "y": [
          4.133792877197266,
          4.162970066070557,
          3.1389613151550293,
          3.173307180404663,
          3.0427145957946777,
          4.086380958557129,
          3.3468480110168457,
          3.0981242656707764
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "cluster=0<br>x=%{x}<br>y=%{y}<extra></extra>",
         "legendgroup": "0",
         "marker": {
          "color": "#EF553B",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "0",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          6.612874507904053,
          7.341375827789307,
          7.228255271911621,
          7.378328800201416,
          7.10992431640625,
          6.773946762084961,
          6.333402156829834,
          6.622437953948975,
          7.169509410858154,
          6.822554588317871,
          6.938701629638672,
          6.417257785797119,
          7.354696273803711,
          7.35639762878418,
          7.207282066345215,
          6.56129789352417,
          6.248010158538818,
          7.083253383636475,
          7.447748184204102,
          7.491954803466797,
          7.039675235748291,
          7.134483337402344,
          6.646735191345215,
          7.240182399749756,
          7.331511974334717,
          6.213351249694824,
          6.980041027069092,
          6.854538440704346,
          6.901212215423584,
          7.580441474914551
         ],
         "xaxis": "x",
         "y": [
          7.405539035797119,
          7.55795955657959,
          7.793301582336426,
          7.622504711151123,
          8.007041931152344,
          8.176390647888184,
          6.724531173706055,
          8.0421142578125,
          8.047597885131836,
          8.339688301086426,
          8.468754768371582,
          6.817569255828857,
          7.754611968994141,
          7.561175346374512,
          7.380877494812012,
          8.466652870178223,
          6.564414978027344,
          8.280616760253906,
          7.486865043640137,
          7.779242992401123,
          7.853209018707275,
          8.029370307922363,
          8.102651596069336,
          8.370865821838379,
          7.9153828620910645,
          6.539876461029053,
          8.437307357788086,
          8.298979759216309,
          7.757956504821777,
          7.761443614959717
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "cluster=1<br>x=%{x}<br>y=%{y}<extra></extra>",
         "legendgroup": "1",
         "marker": {
          "color": "#00cc96",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "1",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          -0.6171921491622925,
          0.11142218112945557,
          -0.5244054794311523,
          -0.6186623573303223,
          -0.7669005990028381,
          -0.4585702419281006,
          -0.49287644028663635,
          -0.5196690559387207,
          -0.6536924839019775,
          -0.7055250406265259,
          -0.6460487246513367,
          -0.49538514018058777,
          -0.5488821268081665,
          0.5688565969467163,
          -0.5144474506378174,
          -0.8251916766166687,
          -0.5974140167236328,
          -0.2020864188671112,
          -0.008523677475750446,
          -0.7231104969978333
         ],
         "xaxis": "x",
         "y": [
          5.8200788497924805,
          5.875269889831543,
          5.763861656188965,
          5.90726900100708,
          6.162840366363525,
          5.825327396392822,
          5.738561153411865,
          6.180556774139404,
          6.293383598327637,
          5.925455093383789,
          6.247088432312012,
          6.515280723571777,
          5.816938877105713,
          6.054632663726807,
          5.654210567474365,
          6.068753242492676,
          6.424282073974609,
          5.765653610229492,
          5.821743011474609,
          6.074626445770264
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "cluster=2<br>x=%{x}<br>y=%{y}<extra></extra>",
         "legendgroup": "2",
         "marker": {
          "color": "#ab63fa",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "2",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          5.134262561798096,
          2.731274366378784,
          5.124255180358887,
          5.362788677215576,
          4.66770601272583,
          1.5416940450668335,
          4.918171405792236,
          3.4040825366973877,
          1.5556756258010864,
          4.1173176765441895,
          4.6828107833862305,
          1.6548224687576294,
          1.5542551279067993,
          2.687469959259033,
          3.502849817276001,
          2.8144161701202393,
          4.396684646606445,
          2.4063796997070312,
          3.167079210281372,
          3.6621110439300537,
          5.397759914398193,
          4.680145740509033,
          1.593687891960144,
          4.867821216583252,
          3.4448885917663574,
          4.0966386795043945,
          4.610622406005859,
          2.824728488922119,
          3.2295303344726562,
          3.1453864574432373,
          3.06618070602417,
          2.688671588897705,
          2.6179516315460205,
          1.6526215076446533,
          2.593902349472046,
          4.17671537399292,
          1.8477883338928223,
          2.400925397872925,
          2.8299062252044678,
          1.7666540145874023,
          3.1793017387390137,
          3.098355293273926,
          5.2806620597839355,
          5.328799724578857,
          2.857872247695923,
          4.692094802856445,
          2.798729419708252,
          4.2916998863220215,
          1.3876954317092896,
          4.415280818939209,
          3.043078899383545,
          3.136510133743286,
          5.2421345710754395,
          2.5175023078918457,
          0.6355592012405396,
          1.495008111000061,
          1.5109953880310059,
          3.0145211219787598,
          3.1022257804870605,
          3.109445333480835,
          4.580254554748535,
          3.839826822280884,
          2.2765796184539795,
          3.028620719909668,
          1.7018054723739624,
          3.206172227859497,
          5.262409210205078,
          1.6391955614089966,
          2.8446044921875,
          2.032188653945923,
          3.1361708641052246,
          5.346282005310059,
          1.620097279548645,
          5.107192516326904,
          1.7353801727294922,
          1.653037190437317,
          2.587716579437256,
          2.5993850231170654,
          2.341114044189453,
          1.9441205263137817,
          2.401519536972046,
          4.066921234130859,
          4.7981791496276855,
          4.71997594833374,
          4.602842807769775,
          5.002500057220459,
          2.6155519485473633,
          3.0664408206939697,
          1.7602027654647827,
          3.521000623703003,
          1.4918220043182373,
          1.6301043033599854,
          5.192516326904297,
          1.6339422464370728,
          3.3441505432128906,
          3.2021446228027344,
          1.862695574760437,
          2.352884292602539,
          2.5451648235321045,
          2.966047763824463,
          4.905935764312744,
          2.244051933288574,
          3.203629970550537,
          3.793182611465454,
          1.9243773221969604,
          2.89936900138855,
          3.120251178741455,
          3.7438268661499023,
          2.2631561756134033,
          1.866214632987976,
          1.9369503259658813,
          2.750985622406006,
          3.158933162689209,
          3.879648208618164,
          2.6532726287841797,
          1.685621738433838
         ],
         "xaxis": "x",
         "y": [
          4.26185417175293,
          6.599162578582764,
          4.352141380310059,
          5.037360191345215,
          4.2622480392456055,
          6.134705543518066,
          4.161435127258301,
          5.841229438781738,
          6.583230018615723,
          5.71846866607666,
          6.522717475891113,
          6.185161113739014,
          6.976951599121094,
          4.139313220977783,
          7.055754661560059,
          4.223972320556641,
          5.278029441833496,
          6.695192337036133,
          7.491102695465088,
          5.595278263092041,
          5.06484317779541,
          4.221990585327148,
          6.4974141120910645,
          6.487144947052002,
          5.507968902587891,
          5.224175453186035,
          4.825448989868164,
          7.370677947998047,
          6.236827373504639,
          6.26958703994751,
          7.36713171005249,
          3.9862632751464844,
          5.035896301269531,
          3.6171064376831055,
          4.882317543029785,
          5.721595287322998,
          5.726563453674316,
          6.652492523193359,
          6.546513080596924,
          3.3877317905426025,
          7.161252975463867,
          6.337097644805908,
          4.198172092437744,
          4.363217353820801,
          5.366308212280273,
          6.329308032989502,
          4.0154500007629395,
          4.255335807800293,
          6.6776347160339355,
          6.1396684646606445,
          4.024847507476807,
          3.982513427734375,
          4.165712356567383,
          4.838964462280273,
          5.856385707855225,
          6.544311046600342,
          6.644408702850342,
          6.895144462585449,
          6.208459854125977,
          5.243352890014648,
          6.60190486907959,
          6.579837322235107,
          3.453509569168091,
          6.904515743255615,
          6.931238651275635,
          7.4333319664001465,
          4.446943759918213,
          6.5969367027282715,
          4.136326313018799,
          7.24819278717041,
          7.467114448547363,
          4.715478897094727,
          6.601465702056885,
          4.302294731140137,
          3.8954336643218994,
          5.993824005126953,
          5.963502883911133,
          5.3571271896362305,
          5.423017978668213,
          7.197818279266357,
          6.563924312591553,
          5.53652811050415,
          4.662449836730957,
          6.3132452964782715,
          5.940138816833496,
          4.195087432861328,
          5.94789457321167,
          6.8696513175964355,
          3.7301125526428223,
          7.277151107788086,
          6.907153606414795,
          7.018045425415039,
          5.299241065979004,
          6.03869104385376,
          6.938735008239746,
          6.72533655166626,
          3.300921678543091,
          7.293379306793213,
          6.396639823913574,
          5.506255626678467,
          4.104763984680176,
          5.51485013961792,
          6.042677879333496,
          5.111448287963867,
          4.028042793273926,
          4.310268402099609,
          5.199843406677246,
          5.570428371429443,
          7.369091987609863,
          3.280663013458252,
          3.1912269592285156,
          6.925230026245117,
          4.164446830749512,
          5.356435298919678,
          7.303502559661865,
          7.109124183654785
         ],
         "yaxis": "y"
        }
       ],
       "layout": {
        "legend": {
         "title": {
          "text": "cluster"
         },
         "tracegroupgap": 0
        },
        "margin": {
         "t": 60
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "white",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "white",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "#C8D4E3",
             "linecolor": "#C8D4E3",
             "minorgridcolor": "#C8D4E3",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "#C8D4E3",
             "linecolor": "#C8D4E3",
             "minorgridcolor": "#C8D4E3",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "white",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "#C8D4E3"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "white",
          "polar": {
           "angularaxis": {
            "gridcolor": "#EBF0F8",
            "linecolor": "#EBF0F8",
            "ticks": ""
           },
           "bgcolor": "white",
           "radialaxis": {
            "gridcolor": "#EBF0F8",
            "linecolor": "#EBF0F8",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "white",
            "gridcolor": "#DFE8F3",
            "gridwidth": 2,
            "linecolor": "#EBF0F8",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "#EBF0F8"
           },
           "yaxis": {
            "backgroundcolor": "white",
            "gridcolor": "#DFE8F3",
            "gridwidth": 2,
            "linecolor": "#EBF0F8",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "#EBF0F8"
           },
           "zaxis": {
            "backgroundcolor": "white",
            "gridcolor": "#DFE8F3",
            "gridwidth": 2,
            "linecolor": "#EBF0F8",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "#EBF0F8"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "#DFE8F3",
            "linecolor": "#A2B1C6",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "#DFE8F3",
            "linecolor": "#A2B1C6",
            "ticks": ""
           },
           "bgcolor": "white",
           "caxis": {
            "gridcolor": "#DFE8F3",
            "linecolor": "#A2B1C6",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "#EBF0F8",
           "linecolor": "#EBF0F8",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "#EBF0F8",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "#EBF0F8",
           "linecolor": "#EBF0F8",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "#EBF0F8",
           "zerolinewidth": 2
          }
         }
        },
        "xaxis": {
         "anchor": "y",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "x"
         }
        },
        "yaxis": {
         "anchor": "x",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "y"
         }
        }
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import plotly.express as px\n",
    "fig = px.scatter(df, x='x', y='y', color='cluster', template='plotly_white')\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "b80b94e7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{-1, 0, 1, 2}"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "set(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "283ede55",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--------------------\n",
      "Cluster 0\n",
      "View and Approve Team Time Off\n",
      "View My Time Off\n",
      "Termination Settings\n",
      "View My Team Timecard\n",
      "View and Approve Time Off Requests\n",
      "Request Time Off\n",
      "View Time Off Requests & Balances\n",
      "View Time Reports\n",
      "View my schedule-web\n",
      "Configure Leave of Absence\n",
      "--------------------\n",
      "Cluster 1\n",
      "Add My Physical Address\n",
      "Send Registration Emails\n",
      "Contact Lifion Support\n",
      "Change Login Methods\n",
      "Edit Birth Details\n",
      "Update my personal email\n",
      "Edit Profile Image\n",
      "Action Link Name\n",
      "Update my marital status\n",
      "Edit My Personal Demographics\n",
      "--------------------\n",
      "Cluster 2\n",
      "Submit a help ticket\n",
      "Import New Hires from ATS\n",
      "View support tickets\n",
      "View Your Tax Statements\n",
      "View Wisely-Flow\n",
      "Integrations Monitoring Dashboard\n",
      "View Your Pay\n",
      "Request Job Change\n",
      "View Legal Entities\n",
      "View Wisely\n",
      "--------------------\n",
      "Cluster -1\n",
      "Core HR Inbound Import Staging List\n",
      "Configure Approval Request Workflow\n",
      "Import Pay Data\n",
      "Work Authorization Type Configuration\n",
      "Associate data import\n",
      "Configure Work Authorization Expiration Reminders\n",
      "Import System Configuration Data\n",
      "Import Legacy System Data\n"
     ]
    }
   ],
   "source": [
    "for i in list(set(labels)):\n",
    "    print(\"--------------------\")\n",
    "    print(f\"Cluster {i}\")\n",
    "    for index in np.where(labels==i)[0][:10]:\n",
    "        print(caption[index])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "faa51814",
   "metadata": {},
   "source": [
    "### Multilingual Action Data Processing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "f9613f8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "from sentence_transformers import util\n",
    "from sentence_transformers import SentenceTransformer\n",
    "\n",
    "# Load model\n",
    "model = SentenceTransformer('/Users/huanglin/Bitbucket/multilingual-raw-data/paraphrase-multilingual-MiniLM-L12-v2')\n",
    "\n",
    "# https://colab.research.google.com/drive/12cn5Oo0v3HfQQ8Tv6-ukgxXSmT3zl35A?usp=sharing#scrollTo=lTKECBnVhwRO\n",
    "def search(inp_question):\n",
    "    start_time = time.time()\n",
    "    question_embedding = model.encode(inp_question, convert_to_tensor=True)\n",
    "    hits = util.semantic_search(question_embedding, embeddings, top_k=5)\n",
    "    end_time = time.time()\n",
    "    hits = hits[0]  #Get the hits for the first query\n",
    "\n",
    "    print(\"Input question:\", inp_question)\n",
    "    print(\"Results (after {:.3f} seconds):\".format(end_time-start_time))\n",
    "    for hit in hits[0:5]:\n",
    "        print(\"\\t{:.3f}\\t{}\".format(hit['score'], caption[hit['corpus_id']]))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "58cd8a30",
   "metadata": {},
   "source": [
    "#### search('Test My Schedule Action Link')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "5b11877f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: 测试我的计划操作链接\n",
      "Results (after 0.138 seconds):\n",
      "\t0.639\tTest My Schedule Action Link\n",
      "\t0.496\tView Criteria Control Center\n",
      "\t0.489\tView my schedule-web\n",
      "\t0.472\tView My Org\n",
      "\t0.465\tView My Schedule\n"
     ]
    }
   ],
   "source": [
    "search('测试我的计划操作链接')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "d63bc210",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: Probar el enlace de acción de mi cronograma\n",
      "Results (after 0.127 seconds):\n",
      "\t0.873\tTest My Schedule Action Link\n",
      "\t0.764\tView my schedule-web\n",
      "\t0.762\tView My Schedule\n",
      "\t0.634\tView Time Off Requests & Balances\n",
      "\t0.608\tView My Time Off\n"
     ]
    }
   ],
   "source": [
    "search(\"Probar el enlace de acción de mi cronograma\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "cc6f0c6c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: Lien d'action Tester mon planning\n",
      "Results (after 0.100 seconds):\n",
      "\t0.560\tForms and plan documents\n",
      "\t0.550\tTest My Schedule Action Link\n",
      "\t0.486\tManage Benefits forms and plan documents\n",
      "\t0.479\tView My Policies\n",
      "\t0.457\tCreate Criteria\n"
     ]
    }
   ],
   "source": [
    "search(\"Lien d'action Tester mon planning\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90a67b69",
   "metadata": {},
   "source": [
    "### Retrieve and Rerank"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "0989c373",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Batches: 100%|██████████| 6/6 [00:00<00:00, 37.37it/s]\n"
     ]
    }
   ],
   "source": [
    "caption = list(set(action_data['caption']))\n",
    "corpus_embeddings = model.encode(caption, convert_to_tensor=True, show_progress_bar=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0694332",
   "metadata": {},
   "source": [
    "#### Retrieve Lexical Search Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "755764d8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 174/174 [00:00<00:00, 790692.20it/s]\n"
     ]
    }
   ],
   "source": [
    "# We also compare the results to lexical search (keyword search). Here, we use \n",
    "# the BM25 algorithm which is implemented in the rank_bm25 package.\n",
    "\n",
    "from rank_bm25 import BM25Okapi\n",
    "from sklearn.feature_extraction import _stop_words\n",
    "from tqdm.autonotebook import tqdm\n",
    "import string\n",
    "\n",
    "\n",
    "# We lower case our text and remove stop-words from indexing\n",
    "def bm25_tokenizer(text):\n",
    "    tokenized_doc = []\n",
    "    for token in text.lower().split():\n",
    "        token = token.strip(string.punctuation)\n",
    "\n",
    "        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:\n",
    "            tokenized_doc.append(token)\n",
    "    return tokenized_doc\n",
    "\n",
    "\n",
    "tokenized_corpus = []\n",
    "for passage in tqdm(caption):\n",
    "    tokenized_corpus.append(bm25_tokenizer(passage))\n",
    "\n",
    "bm25 = BM25Okapi(tokenized_corpus)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "b100c8c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# This function will search all wikipedia articles for passages that\n",
    "# answer the query\n",
    "def search(query):\n",
    "    print(\"Input question:\", query)\n",
    "\n",
    "    ##### BM25 search (lexical search) #####\n",
    "    bm25_scores = bm25.get_scores(bm25_tokenizer(query))\n",
    "    top_n = np.argpartition(bm25_scores, -5)[-5:]\n",
    "    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]\n",
    "    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)\n",
    "    \n",
    "    print(\"Top-5 lexical search (BM25) hits\")\n",
    "    for hit in bm25_hits[0:5]:\n",
    "        print(\"\\t{:.3f}\\t{}\".format(hit['score'], caption[hit['corpus_id']].replace(\"\\n\", \" \")))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "09bf0807",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: Test My Schedule Action Link\n",
      "Top-5 lexical search (BM25) hits\n",
      "\t14.210\tTest My Schedule Action Link\n",
      "\t9.860\tAction Link Name\n",
      "\t4.531\tView My Schedule\n",
      "\t4.178\tTest Time Punch\n",
      "\t3.841\tView My Team Schedule\n"
     ]
    }
   ],
   "source": [
    "search('Test My Schedule Action Link')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "37ff1ff4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: 测试我的计划操作链接\n",
      "Top-5 lexical search (BM25) hits\n",
      "\t0.000\tContact Lifion Support\n",
      "\t0.000\tManage ACA Report Dashboard\n",
      "\t0.000\tManage Benefits forms and plan documents\n",
      "\t0.000\tManage Your Company's Compensation\n",
      "\t0.000\tView My Direct Deposits\n"
     ]
    }
   ],
   "source": [
    "search('测试我的计划操作链接')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "d059b7b0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: Probar el enlace de acción de mi cronograma\n",
      "Top-5 lexical search (BM25) hits\n",
      "\t0.000\tContact Lifion Support\n",
      "\t0.000\tManage ACA Report Dashboard\n",
      "\t0.000\tManage Benefits forms and plan documents\n",
      "\t0.000\tManage Your Company's Compensation\n",
      "\t0.000\tView My Direct Deposits\n"
     ]
    }
   ],
   "source": [
    "search(\"Probar el enlace de acción de mi cronograma\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "8e98e930",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: Lien d'action Tester mon planning\n",
      "Top-5 lexical search (BM25) hits\n",
      "\t0.000\tContact Lifion Support\n",
      "\t0.000\tManage ACA Report Dashboard\n",
      "\t0.000\tManage Benefits forms and plan documents\n",
      "\t0.000\tManage Your Company's Compensation\n",
      "\t0.000\tView My Direct Deposits\n"
     ]
    }
   ],
   "source": [
    "search(\"Lien d'action Tester mon planning\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8cd7010",
   "metadata": {},
   "source": [
    "#### Retrieve Semantic Search Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "e61555e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# This function will search all wikipedia articles for passages that\n",
    "# answer the query\n",
    "def search(query):\n",
    "    print(\"Input question:\", query)\n",
    "\n",
    "    ##### Semantic Search #####\n",
    "    # Encode the query using the bi-encoder and find potentially relevant passages\n",
    "    question_embedding = model.encode(query, convert_to_tensor=True)\n",
    "    # question_embedding = question_embedding.cuda()\n",
    "    hits = util.semantic_search(question_embedding, embeddings, top_k=50)\n",
    "    print(\"Top-5 Semantic search hits\")\n",
    "    hits = hits[0]  # Get the hits for the first query\n",
    "    for hit in hits[0:5]:\n",
    "        print(\"\\t{:.3f}\\t{}\".format(hit['score'], caption[hit['corpus_id']]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "bc85e4d6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: Test My Schedule Action Link\n",
      "Top-5 Semantic search hits\n",
      "\t1.000\tTest My Schedule Action Link\n",
      "\t0.771\tView my schedule-web\n",
      "\t0.750\tView My Schedule\n",
      "\t0.586\tView Team Schedules\n",
      "\t0.583\tView My Team Schedule\n"
     ]
    }
   ],
   "source": [
    "search('Test My Schedule Action Link')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "a7b929a0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: 测试我的计划操作链接\n",
      "Top-5 Semantic search hits\n",
      "\t0.639\tTest My Schedule Action Link\n",
      "\t0.496\tView Criteria Control Center\n",
      "\t0.489\tView my schedule-web\n",
      "\t0.472\tView My Org\n",
      "\t0.465\tView My Schedule\n"
     ]
    }
   ],
   "source": [
    "search('测试我的计划操作链接')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "1803f0a7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: Probar el enlace de acción de mi cronograma\n",
      "Top-5 Semantic search hits\n",
      "\t0.873\tTest My Schedule Action Link\n",
      "\t0.764\tView my schedule-web\n",
      "\t0.762\tView My Schedule\n",
      "\t0.634\tView Time Off Requests & Balances\n",
      "\t0.608\tView My Time Off\n"
     ]
    }
   ],
   "source": [
    "search(\"Probar el enlace de acción de mi cronograma\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "0c7198e8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: Lien d'action Tester mon planning\n",
      "Top-5 Semantic search hits\n",
      "\t0.560\tForms and plan documents\n",
      "\t0.550\tTest My Schedule Action Link\n",
      "\t0.486\tManage Benefits forms and plan documents\n",
      "\t0.479\tView My Policies\n",
      "\t0.457\tCreate Criteria\n"
     ]
    }
   ],
   "source": [
    "search(\"Lien d'action Tester mon planning\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2303fb1c",
   "metadata": {},
   "source": [
    "#### Reranker"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "adcc64e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sentence_transformers import CrossEncoder\n",
    "\n",
    "#The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality\n",
    "# cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')\n",
    "# cross_encoder.save('/Users/huanglin/Bitbucket/multilingual-raw-data/ms-marco-MiniLM-L-6-v2')\n",
    "\n",
    "cross_encoder = CrossEncoder('/Users/huanglin/Bitbucket/multilingual-raw-data/ms-marco-MiniLM-L-6-v2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "bd42b92c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# This function will search all wikipedia articles for passages that\n",
    "# answer the query\n",
    "def search(query):\n",
    "    print(\"Input question:\", query)\n",
    "\n",
    "    ##### BM25 search (lexical search) #####\n",
    "    bm25_scores = bm25.get_scores(bm25_tokenizer(query))\n",
    "    top_n = np.argpartition(bm25_scores, -5)[-5:]\n",
    "    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]\n",
    "    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)\n",
    "\n",
    "    print(\"Top-5 lexical search (BM25) hits\")\n",
    "    for hit in bm25_hits[0:5]:\n",
    "        print(\"\\t{:.3f}\\t{}\".format(hit['score'], caption[hit['corpus_id']].replace(\"\\n\", \" \")))\n",
    "\n",
    "    ##### Semantic Search #####\n",
    "    # Encode the query using the bi-encoder and find potentially relevant passages\n",
    "    question_embedding = model.encode(query, convert_to_tensor=True)\n",
    "    # question_embedding = question_embedding.cuda()\n",
    "    hits = util.semantic_search(question_embedding, embeddings, top_k=50)\n",
    "    print(\"Top-5 Semantic search hits\")\n",
    "    hits = hits[0]  # Get the hits for the query\n",
    "\n",
    "    ##### Re-Ranking #####\n",
    "    # Now, score all retrieved passages with the cross_encoder\n",
    "    cross_inp = [[query, caption[hit['corpus_id']]] for hit in hits]\n",
    "    cross_scores = cross_encoder.predict(cross_inp)\n",
    "\n",
    "    # Sort results by the cross-encoder scores\n",
    "    for idx in range(len(cross_scores)):\n",
    "        hits[idx]['cross-score'] = cross_scores[idx]\n",
    "\n",
    "    # Output of top-5 hits from bi-encoder\n",
    "    print(\"\\n-------------------------\\n\")\n",
    "    print(\"Top-5 Bi-Encoder Retrieval hits\")\n",
    "    hits = sorted(hits, key=lambda x: x['score'], reverse=True)\n",
    "    for hit in hits[0:5]:\n",
    "        print(\"\\t{:.3f}\\t{}\".format(hit['score'], caption[hit['corpus_id']].replace(\"\\n\", \" \")))\n",
    "\n",
    "    # Output of top-5 hits from re-ranker\n",
    "    print(\"\\n-------------------------\\n\")\n",
    "    print(\"Top-5 Cross-Encoder Re-ranker hits\")\n",
    "    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)\n",
    "    for hit in hits[0:5]:\n",
    "        print(\"\\t{:.3f}\\t{}\".format(hit['cross-score'], caption[hit['corpus_id']].replace(\"\\n\", \" \")))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "9e517614",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: Test My Schedule Action Link\n",
      "Top-5 lexical search (BM25) hits\n",
      "\t14.210\tTest My Schedule Action Link\n",
      "\t9.860\tAction Link Name\n",
      "\t4.531\tView My Schedule\n",
      "\t4.178\tTest Time Punch\n",
      "\t3.841\tView My Team Schedule\n",
      "Top-5 Semantic search hits\n",
      "\n",
      "-------------------------\n",
      "\n",
      "Top-5 Bi-Encoder Retrieval hits\n",
      "\t1.000\tTest My Schedule Action Link\n",
      "\t0.771\tView my schedule-web\n",
      "\t0.750\tView My Schedule\n",
      "\t0.586\tView Team Schedules\n",
      "\t0.583\tView My Team Schedule\n",
      "\n",
      "-------------------------\n",
      "\n",
      "Top-5 Cross-Encoder Re-ranker hits\n",
      "\t8.721\tTest My Schedule Action Link\n",
      "\t-3.154\tAction Link Name\n",
      "\t-5.559\tView My Schedule\n",
      "\t-5.697\tView my schedule-web\n",
      "\t-8.245\tView My Team Schedule\n"
     ]
    }
   ],
   "source": [
    "search('Test My Schedule Action Link')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "2568493b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: 测试我的计划操作链接\n",
      "Top-5 lexical search (BM25) hits\n",
      "\t0.000\tContact Lifion Support\n",
      "\t0.000\tManage ACA Report Dashboard\n",
      "\t0.000\tManage Benefits forms and plan documents\n",
      "\t0.000\tManage Your Company's Compensation\n",
      "\t0.000\tView My Direct Deposits\n",
      "Top-5 Semantic search hits\n",
      "\n",
      "-------------------------\n",
      "\n",
      "Top-5 Bi-Encoder Retrieval hits\n",
      "\t0.639\tTest My Schedule Action Link\n",
      "\t0.496\tView Criteria Control Center\n",
      "\t0.489\tView my schedule-web\n",
      "\t0.472\tView My Org\n",
      "\t0.465\tView My Schedule\n",
      "\n",
      "-------------------------\n",
      "\n",
      "Top-5 Cross-Encoder Re-ranker hits\n",
      "\t-6.589\tView Personal Details\n",
      "\t-6.700\tView My Org\n",
      "\t-6.900\tView FAQs\n",
      "\t-7.208\tView Wisely\n",
      "\t-7.273\tCreate Criteria\n"
     ]
    }
   ],
   "source": [
    "search('测试我的计划操作链接')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "d0f1de8f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: Probar el enlace de acción de mi cronograma\n",
      "Top-5 lexical search (BM25) hits\n",
      "\t0.000\tContact Lifion Support\n",
      "\t0.000\tManage ACA Report Dashboard\n",
      "\t0.000\tManage Benefits forms and plan documents\n",
      "\t0.000\tManage Your Company's Compensation\n",
      "\t0.000\tView My Direct Deposits\n",
      "Top-5 Semantic search hits\n",
      "\n",
      "-------------------------\n",
      "\n",
      "Top-5 Bi-Encoder Retrieval hits\n",
      "\t0.873\tTest My Schedule Action Link\n",
      "\t0.764\tView my schedule-web\n",
      "\t0.762\tView My Schedule\n",
      "\t0.634\tView Time Off Requests & Balances\n",
      "\t0.608\tView My Time Off\n",
      "\n",
      "-------------------------\n",
      "\n",
      "Top-5 Cross-Encoder Re-ranker hits\n",
      "\t-10.818\tView Positions\n",
      "\t-10.831\tView FAQs\n",
      "\t-10.842\tView My Org\n",
      "\t-10.935\tClock In/Out\n",
      "\t-10.994\tView Company Org Chart\n"
     ]
    }
   ],
   "source": [
    "search(\"Probar el enlace de acción de mi cronograma\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "2757c841",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input question: Lien d'action Tester mon planning\n",
      "Top-5 lexical search (BM25) hits\n",
      "\t0.000\tContact Lifion Support\n",
      "\t0.000\tManage ACA Report Dashboard\n",
      "\t0.000\tManage Benefits forms and plan documents\n",
      "\t0.000\tManage Your Company's Compensation\n",
      "\t0.000\tView My Direct Deposits\n",
      "Top-5 Semantic search hits\n",
      "\n",
      "-------------------------\n",
      "\n",
      "Top-5 Bi-Encoder Retrieval hits\n",
      "\t0.560\tForms and plan documents\n",
      "\t0.550\tTest My Schedule Action Link\n",
      "\t0.486\tManage Benefits forms and plan documents\n",
      "\t0.479\tView My Policies\n",
      "\t0.457\tCreate Criteria\n",
      "\n",
      "-------------------------\n",
      "\n",
      "Top-5 Cross-Encoder Re-ranker hits\n",
      "\t-9.529\tTest My Schedule Action Link\n",
      "\t-10.745\tView My Org\n",
      "\t-10.804\tView Personal Details\n",
      "\t-10.814\tView Wisely\n",
      "\t-10.829\tManage Settings\n"
     ]
    }
   ],
   "source": [
    "search(\"Lien d'action Tester mon planning\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5d9f3330",
   "metadata": {},
   "outputs": [],
   "source": [
    "### WordToVec Embeddings\n",
    "# import gensim.downloader as api\n",
    "\n",
    "# info = api.info()  # show info about available models/datasets\n",
    "# model = api.load(\"glove-wiki-gigaword-50\")  # download the model and return as object ready for use\n",
    "\n",
    "# model.most_similar(\"request\")\n",
    "# model.most_similar(\"time\")\n",
    "# model.most_similar(\"off\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
