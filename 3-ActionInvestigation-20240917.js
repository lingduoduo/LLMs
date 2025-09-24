{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "07715e3d-1ee3-416d-aba0-6981b09d607f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f04f6916-57c8-4220-a284-9da6962c31ba",
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
       "      <th>details_search_value</th>\n",
       "      <th>docId</th>\n",
       "      <th>caption</th>\n",
       "      <th>category</th>\n",
       "      <th>subtitle</th>\n",
       "      <th>resPos</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Azerbaijan</td>\n",
       "      <td>914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!cbd779b8d...</td>\n",
       "      <td>Bulk Termination</td>\n",
       "      <td>actions</td>\n",
       "      <td>Process bulk termination through an import or ...</td>\n",
       "      <td>49.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Azerbaijan</td>\n",
       "      <td>914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!8fe767bc1...</td>\n",
       "      <td>Manage Broadcast(s) - Voice of the Employee</td>\n",
       "      <td>actions</td>\n",
       "      <td>Manage Broadcast(s) - Voice of the Employee</td>\n",
       "      <td>48.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Azerbaijan</td>\n",
       "      <td>914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!6febe4e80...</td>\n",
       "      <td>Manage Recruiting</td>\n",
       "      <td>actions</td>\n",
       "      <td>Manage Recruiting</td>\n",
       "      <td>47.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Azerbaijan</td>\n",
       "      <td>914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!2b9c08cad...</td>\n",
       "      <td>View My Licenses &amp; Certifications</td>\n",
       "      <td>actions</td>\n",
       "      <td>Manage Licenses or Certification under Certifi...</td>\n",
       "      <td>46.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Azerbaijan</td>\n",
       "      <td>914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!2a1ec3581...</td>\n",
       "      <td>Benefits Data Management</td>\n",
       "      <td>actions</td>\n",
       "      <td>View and edit benefits data, add new records, ...</td>\n",
       "      <td>45.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>538113</th>\n",
       "      <td>thomasb</td>\n",
       "      <td>914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!5d775cc77...</td>\n",
       "      <td>Configure Work Authorization Expiration Reminders</td>\n",
       "      <td>actions</td>\n",
       "      <td>Configure the timing and recipients for notifi...</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>538114</th>\n",
       "      <td>thomasb</td>\n",
       "      <td>914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!7843598ab...</td>\n",
       "      <td>Manage Associates' Pay Data</td>\n",
       "      <td>actions</td>\n",
       "      <td>Maintain and capture Associates' pay data reco...</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>538115</th>\n",
       "      <td>thomasb</td>\n",
       "      <td>914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!af4ef3fa7...</td>\n",
       "      <td>Manage Company Time Off</td>\n",
       "      <td>actions</td>\n",
       "      <td>Manage Time Off Requests &amp; Balances, Policies,...</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>538116</th>\n",
       "      <td>thomasb</td>\n",
       "      <td>914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!4766e3235...</td>\n",
       "      <td>Work Authorization Type Configuration</td>\n",
       "      <td>actions</td>\n",
       "      <td>Manage work visas, permits and other documents</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>538117</th>\n",
       "      <td>thomasb</td>\n",
       "      <td>914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!14c4bd704...</td>\n",
       "      <td>Case Management</td>\n",
       "      <td>actions</td>\n",
       "      <td>This is to enable users to search case managem...</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>538118 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       details_search_value  \\\n",
       "0                Azerbaijan   \n",
       "1                Azerbaijan   \n",
       "2                Azerbaijan   \n",
       "3                Azerbaijan   \n",
       "4                Azerbaijan   \n",
       "...                     ...   \n",
       "538113              thomasb   \n",
       "538114              thomasb   \n",
       "538115              thomasb   \n",
       "538116              thomasb   \n",
       "538117              thomasb   \n",
       "\n",
       "                                                    docId  \\\n",
       "0       914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!cbd779b8d...   \n",
       "1       914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!8fe767bc1...   \n",
       "2       914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!6febe4e80...   \n",
       "3       914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!2b9c08cad...   \n",
       "4       914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!2a1ec3581...   \n",
       "...                                                   ...   \n",
       "538113  914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!5d775cc77...   \n",
       "538114  914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!7843598ab...   \n",
       "538115  914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!af4ef3fa7...   \n",
       "538116  914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!4766e3235...   \n",
       "538117  914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!14c4bd704...   \n",
       "\n",
       "                                                  caption category  \\\n",
       "0                                        Bulk Termination  actions   \n",
       "1             Manage Broadcast(s) - Voice of the Employee  actions   \n",
       "2                                       Manage Recruiting  actions   \n",
       "3                       View My Licenses & Certifications  actions   \n",
       "4                                Benefits Data Management  actions   \n",
       "...                                                   ...      ...   \n",
       "538113  Configure Work Authorization Expiration Reminders  actions   \n",
       "538114                        Manage Associates' Pay Data  actions   \n",
       "538115                            Manage Company Time Off  actions   \n",
       "538116              Work Authorization Type Configuration  actions   \n",
       "538117                                    Case Management  actions   \n",
       "\n",
       "                                                 subtitle  resPos  \n",
       "0       Process bulk termination through an import or ...    49.0  \n",
       "1             Manage Broadcast(s) - Voice of the Employee    48.0  \n",
       "2                                       Manage Recruiting    47.0  \n",
       "3       Manage Licenses or Certification under Certifi...    46.0  \n",
       "4       View and edit benefits data, add new records, ...    45.0  \n",
       "...                                                   ...     ...  \n",
       "538113  Configure the timing and recipients for notifi...     4.0  \n",
       "538114  Maintain and capture Associates' pay data reco...     3.0  \n",
       "538115  Manage Time Off Requests & Balances, Policies,...     2.0  \n",
       "538116     Manage work visas, permits and other documents     1.0  \n",
       "538117  This is to enable users to search case managem...     0.0  \n",
       "\n",
       "[538118 rows x 6 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "actions = pd.read_csv(\"actions_df_de5c290a-e50f-406a-8efc-1faaca882022.csv\")\n",
    "actions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6e8ee6da-4845-448f-b9cc-4753b5a2773b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 538118 entries, 0 to 538117\n",
      "Data columns (total 6 columns):\n",
      " #   Column                Non-Null Count   Dtype  \n",
      "---  ------                --------------   -----  \n",
      " 0   details_search_value  538118 non-null  object \n",
      " 1   docId                 532984 non-null  object \n",
      " 2   caption               538118 non-null  object \n",
      " 3   category              538118 non-null  object \n",
      " 4   subtitle              538118 non-null  object \n",
      " 5   resPos                532984 non-null  float64\n",
      "dtypes: float64(1), object(5)\n",
      "memory usage: 24.6+ MB\n"
     ]
    }
   ],
   "source": [
    "actions.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8c7f5777-2935-4e79-b07d-9be9dac1b7e8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "details_search_value                                           Azerbaijan\n",
       "docId                   914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!cbd779b8d...\n",
       "caption                                                  Bulk Termination\n",
       "category                                                          actions\n",
       "subtitle                Process bulk termination through an import or ...\n",
       "resPos                                                               49.0\n",
       "Name: 0, dtype: object"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "actions.iloc[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "21e9135d",
   "metadata": {},
   "outputs": [],
   "source": [
    "actions['locale'] = actions.docId.apply(lambda x: str(x).split(\"!\")[-1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "40315f10",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "locale\n",
       "en-US     496758\n",
       "en-GB      35926\n",
       "nan         5134\n",
       "fr-CA        150\n",
       "es-001       100\n",
       "en-AU         50\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "actions.groupby('locale').size().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ce4833c1-e112-40b1-8af9-b583cc64e059",
   "metadata": {},
   "source": [
    "### Checking Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3e053b10-1cdf-4b65-8c0e-b8a9bfea1ad7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def find_all_missing_columns(df: pd.DataFrame) -> list:\n",
    "    all_missing_cols = []\n",
    "    \n",
    "    for column in df.columns:\n",
    "        non_missing_count = df[column].notnull().sum()\n",
    "        is_all_missing = (non_missing_count == 0)\n",
    "        if is_all_missing:\n",
    "            print(f\"All values in '{column}' are missing: {is_all_missing}\")\n",
    "            all_missing_cols.append(column)\n",
    "    \n",
    "    return all_missing_cols"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "4621e122-9130-4d6a-8dd3-5b94ae19768c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "find_all_missing_columns(actions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "3b538ed7-236d-4f70-ac09-7f9d21879c9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def calculate_missing_data(df: pd.DataFrame) -> dict:\n",
    "    # Calculate the total number of rows in the DataFrame for percentage calculation\n",
    "    total_rows = len(df)\n",
    "    missing_data = {}\n",
    "    \n",
    "    for column in df.columns:\n",
    "        # Count the number of non-missing (nonnull) entries in the column\n",
    "        non_missing_count = df[column].notnull().sum()\n",
    "        missing_count = total_rows - non_missing_count\n",
    "        missing_percentage = (missing_count / total_rows) * 100\n",
    "        if 0 < missing_percentage < 100:  \n",
    "            missing_data[column] = {'missing_count': missing_count, 'missing_percentage': missing_percentage}\n",
    "            print(f\"Column '{column}' has {missing_count} missing rows, which is {missing_percentage:.2f}% of the total rows.\")\n",
    "    \n",
    "    return missing_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "d0a13b4f-6fb1-4eac-ae13-86cb3816cec8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Column 'docId' has 5134 missing rows, which is 0.95% of the total rows.\n",
      "Column 'resPos' has 5134 missing rows, which is 0.95% of the total rows.\n"
     ]
    }
   ],
   "source": [
    "res = calculate_missing_data(actions)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f7fc39be-e8dc-4942-8222-3865d1d68302",
   "metadata": {},
   "source": [
    "### Split the Features into Different Categories"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "77c6d7c3-821e-414f-92e6-b0c634e89b6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def identify_columns(df: pd.DataFrame) -> dict:\n",
    "    identifier_cols = [col for col in df.columns if '_id' in col.lower()]\n",
    "    print(f\"Identifier cols: {identifier_cols}\")\n",
    "    \n",
    "    categorical_cols = [col for col in df.columns if df[col].dtype in ['object', 'category', 'bool'] and col not in identifier_cols ]\n",
    "    print(f\"Categorical cols: {categorical_cols}\")\n",
    "\n",
    "    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in identifier_cols ]\n",
    "    print(f\"Numeric cols: {numeric_cols}\")\n",
    "    \n",
    "    return {\n",
    "        'identifier_cols': identifier_cols,\n",
    "        'categorical_cols': categorical_cols,\n",
    "        'numeric_cols': numeric_cols\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "d55f65a9-db05-4229-8410-897c35fc50c9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Identifier cols: []\n",
      "Categorical cols: ['details_search_value', 'docId', 'caption', 'category', 'subtitle', 'locale']\n",
      "Numeric cols: ['resPos']\n"
     ]
    }
   ],
   "source": [
    "cols = identify_columns(actions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5a36c4bb-c2ef-4577-9012-8f0928a532e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['details_search_value', 'docId', 'caption', 'category', 'subtitle', 'locale']"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "c_cols = cols['categorical_cols']\n",
    "c_cols"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b39871f4-3365-4116-82a8-e7f865480618",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['resPos']"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "n_cols = cols['numeric_cols']\n",
    "n_cols"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2d285446-b533-400e-a57d-ff68424f662e",
   "metadata": {},
   "source": [
    "### Categorical Data EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c745b048-7132-4338-a773-e02041cf8260",
   "metadata": {},
   "outputs": [],
   "source": [
    "def categorical_feature_distribution(df: pd.DataFrame, categorical_cols) -> dict:\n",
    "    distribution_counts = {}\n",
    "    \n",
    "    for col in categorical_cols:\n",
    "        print(\"----------------------------------------\")\n",
    "        distribution = df.groupby(col).size().sort_values(ascending=False)\n",
    "        distribution_counts[col] = distribution\n",
    "        print(f\"Distribution for '{col}':\")\n",
    "        print(distribution)\n",
    "        print(\"\\n\\n\")\n",
    "    \n",
    "    return distribution_counts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "8eb472a8-41d5-430b-8ca0-48b49d6f3ada",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "----------------------------------------\n",
      "Distribution for 'details_search_value':\n",
      "details_search_value\n",
      "Test                     22305\n",
      "All Delegates            15382\n",
      "View People Movements    13238\n",
      "manage                   10432\n",
      "Teams                     9922\n",
      "                         ...  \n",
      "43132                        1\n",
      "242522                       1\n",
      "242032                       1\n",
      "242031                       1\n",
      "timw                         1\n",
      "Length: 4050, dtype: int64\n",
      "\n",
      "\n",
      "\n",
      "----------------------------------------\n",
      "Distribution for 'docId':\n",
      "docId\n",
      "002!411afb6ee5534505a3da0498a526c0c7!global!en-US                                     3135\n",
      "002!a470dfce66d5470686573a5c00e28e42!global!en-US                                     2927\n",
      "002!c6781f866a2a47e9b7a0ad6331d51f91!global!en-US                                     2876\n",
      "002!365b3587a07c43d7b4ae6208a82c748f!global!en-US                                     2410\n",
      "002!953b85895c60422da443819f357f7e3b!global!en-US                                     2360\n",
      "                                                                                      ... \n",
      "bc25929c-d016-4d0a-a040-503cb0724217!0f75ebb429d041ec8da0b1bfb7215804!global!en-US       1\n",
      "002!e690d0fbd9844910bed5fd0f2d7c9b81!global!fr-CA                                        1\n",
      "5e0055f3-b96d-410b-8bbb-2911e88e15af!60fe62786040470c96d9ae79450dcd2c!global!en-US       1\n",
      "002!eddd2073875440588e36020ee80d15f6!global!fr-CA                                        1\n",
      "914413dd-9bb0-40fa-8aa6-7c4ebed0bd28!d8c0511ea839407e89067c6959dc448d!global!en-AU       1\n",
      "Length: 3062, dtype: int64\n",
      "\n",
      "\n",
      "\n",
      "----------------------------------------\n",
      "Distribution for 'caption':\n",
      "caption\n",
      "Manage Associates' Pay Data                  8217\n",
      "Manage Pay Runs                              8145\n",
      "View People Movements                        7133\n",
      "Manage Benefits Supplemental Fields          7061\n",
      "Request Time Off                             6932\n",
      "                                             ... \n",
      "Ver mis identificaciones                        1\n",
      "Gestion des actifs                              1\n",
      "Gestion des attributions                        1\n",
      "Gestion des données des avantages sociaux       1\n",
      "Étalonner les attributs du talent               1\n",
      "Length: 469, dtype: int64\n",
      "\n",
      "\n",
      "\n",
      "----------------------------------------\n",
      "Distribution for 'category':\n",
      "category\n",
      "actions    538118\n",
      "dtype: int64\n",
      "\n",
      "\n",
      "\n",
      "----------------------------------------\n",
      "Distribution for 'subtitle':\n",
      "subtitle\n",
      "Maintain and capture Associates' pay data records within a legal entity and/or pay group.            8217\n",
      "Process payroll data within a pay group for a given pay period and to review and commit Pay Runs.    8145\n",
      "View a summary of recent new hires, job changes and terminations                                     7133\n",
      "View and Manage Associate's benefit related supplemental eligibility and compliance values.          7061\n",
      "View Your Benefits                                                                                   6895\n",
      "                                                                                                     ... \n",
      "Gérer les motifs et les catégories de congés                                                            1\n",
      "Gérer les enregistrements I9 des collaborateurs                                                         1\n",
      "Gérer les accommodements spéciaux                                                                       1\n",
      "Gérer le processus d'entrevues de départ de votre organisation                                          1\n",
      "Étalonner les attributs du talent                                                                       1\n",
      "Length: 484, dtype: int64\n",
      "\n",
      "\n",
      "\n",
      "----------------------------------------\n",
      "Distribution for 'locale':\n",
      "locale\n",
      "en-US     496758\n",
      "en-GB      35926\n",
      "nan         5134\n",
      "fr-CA        150\n",
      "es-001       100\n",
      "en-AU         50\n",
      "dtype: int64\n",
      "\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "res = categorical_feature_distribution(actions, c_cols)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "a0165887-34ea-4f70-b425-3973bd2715d4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "details_search_value\n",
       "Test                     22305\n",
       "All Delegates            15382\n",
       "View People Movements    13238\n",
       "manage                   10432\n",
       "Teams                     9922\n",
       "                         ...  \n",
       "43132                        1\n",
       "242522                       1\n",
       "242032                       1\n",
       "242031                       1\n",
       "timw                         1\n",
       "Length: 4050, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "actions.groupby(actions['details_search_value']).size().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "7da092ec-5410-4d7e-9dc1-b36f3e299599",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "caption\n",
       "Manage Associates' Pay Data                  8217\n",
       "Manage Pay Runs                              8145\n",
       "View People Movements                        7133\n",
       "Manage Benefits Supplemental Fields          7061\n",
       "Request Time Off                             6932\n",
       "                                             ... \n",
       "Ver mis identificaciones                        1\n",
       "Gestion des actifs                              1\n",
       "Gestion des attributions                        1\n",
       "Gestion des données des avantages sociaux       1\n",
       "Étalonner les attributs du talent               1\n",
       "Length: 469, dtype: int64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "actions.groupby(actions['caption']).size().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "76655d67",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "locale\n",
       "en-US     496758\n",
       "en-GB      35926\n",
       "nan         5134\n",
       "fr-CA        150\n",
       "es-001       100\n",
       "en-AU         50\n",
       "dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "actions.groupby(actions['locale']).size().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f2f8e511",
   "metadata": {},
   "source": [
    "### Sentence Transformer Embeddingsm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "da05cd5e-494b-4dc3-8d2a-1963213e437d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/huanglin/miniconda3/envs/python11/lib/python3.11/site-packages/sentence_transformers/cross_encoder/CrossEncoder.py:11: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from tqdm.autonotebook import tqdm, trange\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(3, 384)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# After that you should be able to just use this code to load the model even on the internal network\n",
    "from sentence_transformers import SentenceTransformer\n",
    "# Load model\n",
    "model = SentenceTransformer('/Users/huanglin/Bitbucket/xinpeng-multilingual-raw-data/paraphrase-multilingual-MiniLM-L12-v2')\n",
    "\n",
    "# Convert text to text embeddings\n",
    "sentence = ['Request Time Off', 'View My Policies', 'Manage Fluid Field Definitions']\n",
    "model.max_seq_length = 384\n",
    "vector = model.encode(sentence)\n",
    "vector.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "949fdf7c",
   "metadata": {},
   "source": [
    "### Semantic Textual SImilarity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "eb99a169",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "469"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "caption = list(set(actions['caption']))\n",
    "len(caption)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "39110d9e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Batches: 100%|██████████| 15/15 [00:01<00:00, 12.00it/s]\n"
     ]
    }
   ],
   "source": [
    "embeddings = model.encode(caption, show_progress_bar=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "155b298f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(469, 384)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "embeddings.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04f1e5e8",
   "metadata": {},
   "source": [
    "### Reducing the Dimensionality of Embeddings\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "07c402d7",
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
   "execution_count": 25,
   "id": "b8e75a8f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(469, 5)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reduced_embeddings.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "48d4bd20",
   "metadata": {},
   "source": [
    "### Cluster the Reduced Embeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "bd3822a6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(469,)"
      ]
     },
     "execution_count": 30,
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
   "execution_count": 31,
   "id": "1dacf364",
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
   "execution_count": 32,
   "id": "e21b4390",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{-1, 0, 1, 2, 3, 4, 5, 6}"
      ]
     },
     "execution_count": 32,
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
   "execution_count": 33,
   "id": "a3e4233a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='x', ylabel='y'>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "\n",
    "sns.scatterplot(data=df, x='x', y='y', hue='cluster', linewidth=0, legend=False, s=30, alpha=0.9)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "86f60521",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Select outliers and non-outliers (clusters)\n",
    "to_plot = df.loc[df.cluster != \"-1\", :]\n",
    "outliers = df.loc[df.cluster == \"-1\", :]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "049f8cd3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='x', ylabel='y'>"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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LViwQAMGDNCSJUv04x//WLfccouefvpps6N1Ca+88oqOHDmi66+/3uwolnb33Xfr2muv1eDBg+VyuTRq1CjddtttmjlzptnRLCktLU1nn3227r//flVUVCgUCmnhwoX68MMPVVlZaXY8y2H6KY4cOXJERUVFmjdvnm666Saz41hOQkKCxo4dq5UrV0a23XLLLVq1apU+/PBDE5N1DZdeeqkSEhL0+uuvmx3F0p5//nndddddevjhhzV06FCtW7dOt912m+bNm6dZs2aZHc+SduzYoRtvvFHLly+Xw+HQ6NGjNXDgQK1Zs0ZbtmwxO56lsOo0jmRkZGjgwIEqLS01O4ol5eXlaciQIc22nXXWWfr73/9uUqKuY/fu3Vq6dKleeukls6NY3l133RUZrZGk4uJi7d69W3PnzqXUdJJ+/frpvffeU11dnXw+n/Ly8jR9+nT17dvX7GiWw/RTHKmtrdWOHTuUl5dndhRLOuecc7Rt27Zm27Zv366ioiKTEnUdTz75pHJycpotpETnqK+vl93e/J9+h8OhcDhsUqKuIyUlRXl5eTp8+LCWLFmiq666yuxIlsNITQy78847NXXqVBUVFamiokL33nuvHA6HZsyYYXY0S7r99ts1ceJEPfjgg/rWt76lTz75RI899pgee+wxs6NZWjgc1pNPPqlZs2ZxyYIomDp1qh544AH16tVLQ4cO1aeffqp58+bpxhtvNDuaZS1ZskSGYWjQoEEqLS3VXXfdpcGDB+uGG24wO5r1GIhZ06dPN/Ly8oyEhASjoKDAmD59ulFaWmp2LEt7/fXXjWHDhhlut9sYPHiw8dhjj5kdyfKWLFliSDK2bdtmdpQuwefzGbfeeqvRq1cvIzEx0ejbt6/xH//xH4bf7zc7mmW98MILRt++fY2EhASjR48exuzZs40jR46YHcuSWCgMAAAsgTU1AADAEig1AADAEig1AADAEig1AADAEig1AADAEig1AADAEig1AADAEig1AADAEig1AADAEig1AADAEig1AADAEig1AOLWgQMH1KNHDz344IORbStXrlRCQoKWLVtmYjIAZuCGlgDi2uLFi3X11Vdr5cqVGjRokEaOHKmrrrpK8+bNMzsagCij1ACIe7Nnz9bSpUs1duxYbdiwQatWrZLb7TY7FoAoo9QAiHsNDQ0aNmyYysvLtWbNGhUXF5sdCYAJWFMDIO7t2LFDFRUVCofD2rVrl9lxAJiEkRoAca2pqUnjx4/XyJEjNWjQIM2fP18bNmxQTk6O2dEARBmlBkBcu+uuu7Ro0SKtX79eqampOv/88+XxePTGG2+YHQ1AlDH9BCBuvfvuu5o/f76eeeYZpaeny26365lnntGKFSu0YMECs+MBiDJGagAAgCUwUgMAACyBUgMAACyBUgMAACyBUgMAACyBUgMAACyBUgMAACyBUgMAACyBUgMAACyBUgMAACyBUgMAACyBUgMAACzh/wN7n1Yz4wTG4QAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(data=outliers, x='x', y='y', hue='cluster', linewidth=0, legend=False, s=30, alpha=0.9)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "56fa020c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='x', ylabel='y'>"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjUAAAGwCAYAAABRgJRuAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAACjr0lEQVR4nOzdd3xb5fX48c+9Wt7bznT2InuHJKxAGGHvPcrsAEqh7Zem/XXw7belFDqBQqEUKJuyCTOMDCBk772XHTvxXlr33t8fj21JtuTYiWXJ9nn3pWLdq/HISaSj5znPOZplWRZCCCGEEJ2cHusBCCGEEEK0BwlqhBBCCNElSFAjhBBCiC5BghohhBBCdAkS1AghhBCiS5CgRgghhBBdggQ1QgghhOgS7LEeQEcyTZOCggJSU1PRNC3WwxFCCCFEK1iWRVVVFb1790bXI8/HdKugpqCggPz8/FgPQwghhBDHYP/+/fTt2zfi+W4V1KSmpgLql5KWlhbj0QghhBCiNSorK8nPz2/8HI+kWwU1DUtOaWlpEtQIIYQQnczRUkckUVgIIYQQXYIENUIIIYToEiSoEUIIIUSXIEGNEEIIIboECWqEEEII0SVIUCOEEEKILkGCGiGEEEJ0CRLUCCGEEKJLkKBGCCGEEF2CBDVCCCGE6BK6VZsEIYToSEt3lfDq8v3sKamhT0YiV03J5+Shua267/oDFep+mYlM7JcZ5ZEK0TVIUCOEEFHw3toCfvfBJiwLDNNif2kty3aX8uOzhnHVlH4R71dS7eF/3ljH+oMVjceG9kjhT1eMp2d6QkcMXYhOS5afhBCint8w+XxzEQ9+uJk/z9/GhqDAoi28fpPHvthOrdegpNpDaY2XkmovFXU+Hv1iBzUef8T7/ub9TSEBDcD2omp+9ta6YxqLEN2JzNQIIQRQ5zX44aurWbu/vPHYq8v2ccOJ/bn7jKHNbl9c6eat1QfZdbiaXumJXDKhDwNykgHYUFBBUaWHarcfK+g+Hr9JYXkd3+4q4YwTejR7zP2ltSzdVRJ2fJsKKtlcWMkJvdKO63UK0ZXJTI0QQgAvfLsnJKAJHN/LmibH1x+o4OqnvuXfX+1mwdbDvLJsH9f9aylfbCkCwKHrVLt9IQFNA8OC1fuaPw9AUaW7xTEeOsp5Ibo7CWqEEAL4eMOhVp/73YebqG6yhOQzTH73wWbcPoNheckY4SKaesv3loY9PiAnGbuuRbxfncfPugPlmGYLDy5ENybLT0KIbqnK7eOpRbtYtO0wSU47hyrcWJaFpjUPKmq9gQBmR3E1uw7XNLuNYVoUlNdx0kNfYLe1/H3xYFld2OM5KS7mjOnF+2sLQo67fQYWKt8GoFd6Ar84byRTB2Yd7WUK0a10qpmagwcPcv3115OdnU1iYiJjxoxhxYoVsR6WEKKTKa5yc/ojC3n8yx2sP1jB0t0lFFa4qajzY1nNZ0GmDAgED16/2ey8aVlU1Pnw+E18hnXUmZT0BEfEc/efM4LLJvXFaVdvz4Zp4TMsEuw6lmXhM0z2ltTyo9dWc7A8fHAkRHfVaWZqysrKmDlzJrNmzeKjjz4iNzeX7du3k5kp9RuEEG1z72trKKnxhB7UwGuYePwmCQ5b4+GhPVI4a1SPkOs5KS6OVAfu7/YZmPXBkMOmNf7XF2YNStfgtBGhtWqKKt3MW1dISbWHEb3SuOeMofzgtMEUVrh55JOtrNlfjs8wqXL7G5+n0u3jp/9dy0u3TQs7uyREd9RpgpqHHnqI/Px8nn322cZjAwcOjOGIhBCd1co9Zc2OafX/l+CwkeS0keCwcdaontx28kBc9kCQ47Dp3DlrCP87byMNkzoNwUuS045eH2CkJjgoq/FiNTw2YNM18tJcXDM1UKfmiy1F/PKdjfiMwAzQ04t2ccaIPDYVVvLt7hIs06LWazRLPF61r4xXlu3n2mmR694I0Z10mqDmvffe4+yzz+aKK65g4cKF9OnThx/84AfcfvvtEe/j8XjweALfpiorKztiqEKIOGaYFv4Iy0MaKl/l3btOavExzhvbi+wUJy8v3cfOw9XYbTo1Hj8ue2BF365rZCSpZSbTVLk6E/tlcN9Zw+mbmQSovJ4H3t8UEtCYlsWWQ5VsL6oiLdGBx2fi9ZtYqFmeYDZN47UVEtQI0aDTBDW7du3iiSee4L777uPnP/85y5cv54c//CFOp5Obbrop7H0efPBBHnjggQ4eqRAinjXMlhyqCL89esbgnFY9zomDsjlxUDYAi7Yd5if/XdvsNg6bzi/OO4Hpg7LRNI3cVFfI+QVbD1PnNUKO1XkNDNPCwMK0LBIdNjz1eTyWBZpG4wyRy65TWF6HYVrYWtg1JUR30WkShU3TZOLEifz+979nwoQJ3HHHHdx+++08+eSTEe8zd+5cKioqGi/79+/vwBELIeLVPWcMbVwmCpae6ODuM4a0+fFOGZbLFZP7Njs+e2QPzh/bm7y0hGYBDRC2srAnKBHZssBp10lwqLdqq/6YBaBBRZ0fXQO/2Tx5WYjuqNPM1PTq1YuRI0eGHDvhhBN48803I97H5XLhcjV/IxFCdG9XTemH1zB59PMdlNV60TWNE3ql8vdrJpDkPLa3xZ+ePYLzxvTmy63FGKbFyUNzmHCURpST+kc+r2ta43JTisuOz/ABanmq4duohUWN1+CB9zfx+0vGHNO4hehKOk1QM3PmTLZu3RpybNu2bfTv3z9GIxJCdCSP3+DLLYc5WF7HwJwkThmae9R6MC254cQBXDu1PwXldSS77GQlO497jCN7pzGyd/g2BgXldRRXeRiQnURGknquoT1SmT2yB59tKmq8ndOu4/YZJDttjbuadE0jzWWj2ms0Lj1paCQ6bSQ6bHy+uYh9pwymX3bScb8GITqzThPU3HvvvcyYMYPf//73XHnllSxbtoynnnqKp556KtZDE0JEyf7SWt5ZfZAthypZsacMw7TQ66cv+mYm8ui1E+mTkXjMj2/TNfKzohsIlNZ4eeD9jXy7qwTDtHDYdC6Z0Id7zxyGw6bzwIWjGJSTzDtrDlJS7WVCfgYFFe5mS1Nj8zPYXFiF37SwLAu7TW/cVWVZsKmwUoIa0e1pVrhKU3Fq3rx5zJ07l+3btzNw4EDuu+++Fnc/NVVZWUl6ejoVFRWkpUlTOCHi2YKtxfzi7Q34DJOyWi9G/Q6itAQ7jvoZmrF9M/jXTZNjPNKW3fLcclbvK6Pa48eo33Vl13UundiHP1w2Nux9ymq8/HflfpbsLMFlt3HmyB5MGZDJ5U8uabyNzzAbCwE67TpP3Tg5pEigEF1Jaz+/O1VQc7wkqBGic/D4Dc7/+1dU1PnwGSYVdb7GczZdIzMpsFT03+9Np392ciyGeVRr95dz83PLKa/1Njunaxof/vAkhvVs/XvRnS+vYvnuUqrcfjz+wK4pXdO4Zmo+D1w4unEmS4iupLWf351m95MQovtYuqu0MZBpWlJG1ZkJ7PYprWkeMMSL3Udqmm3ZbmBaFi8u3demx/v1BSPJTHI2C2jSEux8srGIj1poyilEd9BpcmqEEN1H8LbmhrYDwRrmlxMcNobkpXTUsI6BhcdvYFqqsF/TXeSFFeF7N+0tqWHhtsOYpsUpw3IZlKteY15qAif0SqWo0o3ftLDp4LTpjQnFH64v5LyxvaL5goSIaxLUCCHizqT+mThsOj7DbMylachH0TVVrRfgysl9SW2hOWRH21tSw9LdpSQ4bBimyUMfbWkMwBpqzGiaCnBcdhu90lWSc7XHz4frC1l/oILtxVVsL6rCpquJ9H8s2MlVU/L58VnD629r4LTrhNurVRWm7o0Q3YkENUKIuJOV7OTG6f15YuFOKuuXoTQCgYHftLjnjCHcNGNALIfZyDQtHvxoM++tLcCy1NJSea2PZJedFJeNKk9guciyVCXgFJeNiyf0obCiju+9uIrC8jq8htn4epNddhLrG2u+tnw/Y/tmcPLQHIbkJbP2QDnhMmemtFD3RojuQIIaIURc+u6pg3lj5QFqPaoDts2mkWDXcdh0NA1OGpoTN92p31h1gHfXFDRe9/lNTMuiyu0jM8lJMlAbFNikJzn44RlDmdgvk5+9uY7CcrUM5fYFblPj8eO06Y3tDx75ZAu/+8Ck2u2nwu3DYdNJdtkbg5vMJAc90xN4/ps99MpIYGzfdHqkJsTN70iIjiBBjRAiLh2u8lBR52tsCtnUt7tKGZKX2sGjCu+d1QdDrgfnNrt9BskuOwl2G17DRAOevH4SE/pl4vYZLNp2uPG2TZOivX4Tl12nwu3jcJUHp13HZddJddmp8xnUeQ0ykx2M65PB1qIqHvp4CzUePx6/iU3XGNEzlTtnDeGc0ZJnI7oH2f0khIhLLocetj9TgySnrQNH07LDVZ6Q646gSscNgYpN10h02BiUm8K4vhmAqjUT3DHc0WQ7tmFZlNX58BlW4+2rPX5qfQYpLjspLjuv3H4iByvqqKjzUVHna0yyNkyLbUXV/OrdjXy+uQghugMJaoQQcSktwcH0wdlhzzlsOqePyOvgEUU2oldo3YyGAAbAHrR7S9c07pw1pLGWTGqCI6StQmJQawRQgYkZ6IvQyOs38Rnq3EfrD3GwrA6P38BvWJgWjRevXyVaP/fNnnZ+xULEJwlqhBBx674zh5GXFtqUVtc07p8zorF/Ujy4cXr/ZrNKyS47vdITGdc3g+wUF9MHZ/PotRM4c2SPkNvdOWtI48yOrmlkJDpw2NQyk9+0cOh6446pYF6jftu7poKeGo9B00qqFmp7/NZDVfgN6eQtuj7JqRFCxK38rCReuf1EPlp/iE2FlWQnOzl/XG8G5sRXBeEpA7J48LIx/OPLHewtqUXTYHL/LP7nnOFHrXY8ZUAWT1w/kee+2cP6AxVkJjm5YFwvLhrfm4se/4Yaj7+xsnJTw3umcvbIXjyxYFfjlvem3D4/vTMSjqv5pxCdhQQ1Qoi4lprg4Mop+bEexlHNGp7HacNyKaxw47LrZKe4jn6nemP7ZvDnK8eHfcx56wpIcdkorzNpKKSsadAvK4nfXzKG/KwkBmQnUV7rpWnTGw21DHXOqJ7H/sKE6EQkdBdCiHaiaRq9MxLbFNC05PZTBuJy6JTX+VXAoqmA5oSeaXz4w5MbO4xfMrEPCQ4bwXnGDT+6HDZuO3lgu4xHiHgnQY0QQoRRVuPl/bUFvLvmYLPdTR2lss6P12eS6LBht+k4bTqpCQ6O1HhYta+s8XbnjOpFZpKT7GQn2UlOUl02klw2Ehw2MhMd/L93NvLxhkN0o/7FopuS5SchhGjiteX7+PvnOxrzWOy6xs0zB3L7KYM6dBz/XbEfC7V9PYnAFnbLUlWGJw/IAiA31cX9c0bw+w82o2kWTs1GeZ0PHbUD6ttdJXy7q4Tle0r55fkjO/Q1CNGRZKZGCCGCrNlfzp8+3RaSmOs3LZ5evIsFW4s7dCwHywMNL/2mRWWdjyPVHo5Ue1i8/UjIDNKF43rz3+9N55aTBpKb4iLZaSMjydG4fRzg/bUFbDhY0aGvQYiOJEGNEEIEeXvVgcjnmlQOjrZ+2SpnxjAtKmp9gW3cQHmdlzteWEGV29d4LD8riTtOHkSl20eCwxa2RcLCoArGQnQ1EtQIIUSQw9WR82eKKzs2t+aKSX1x2HRqvQZWkyo0iQ4bB8vqmLeusPGY12/y7pqDlNX6qKzz4fYbTR9SiC5NghohhAgytIV+UsN7tl+vqT1Hapi3roBvdhyJWBhvSF4qD146huBuUpqmkeKyNxbsW12fMOz2Gdz58ioe/GgLFhZeo775ZZ0v5DFPHZbbbq9BiHgjicJCCBHkisl9eXfNQWq9obMcDpvONVP7Hffje/0mv34vtB9Tj7QE/nDZGEb1Tm92+1OG5TI+P4Mth6oAlbQcvKyU4lINP99efZC1+8sBSHba8Rk+LMvCZ5i4fQYJDhsXT+jD6D7Nn0OIrkJmaoQQIkjfzCQevWZiSD+nQbnJ/OnKce0yU/OPBTuaNZgsqnRz32trcfvCLxedN7Y3DpuOw6Y3y5M5b6zqwP355kASs01X7RYSHTbsuk5Kgp3fXTKGuXNGHPf4hYhnMlMjhBBNjOmbzn9umUpBeR2GaTUWuTtePsPkvTUFYc+V1Xr5Yksx547p1ezcVVPyWbWvjK+2Hwk5/p2ZA5jUPxMg0Piynk3XSHapt/ixfTKa9ZwSoiuSoEYIISLonZHYro9X7fZT7fFHPF9Y4Q573GHT+fOV41m+p5QlO0tw2HTOHJlHflYSGwsqSHU5OGlITsTt2icNzWmX8QsR7ySoEUKIDpKW6CAnxcWRCDushualtHj/KQOymFJfcO+VZfv43ourqKxPBB7RM5We6QkcahIYDclL4ZIJfdph9ELEP8mpEUKIDmLTNa6dFj7ZeEBOMjOHtG5G5YN1hfxl/rbGgAZgy6EqPH6TG6f3Z0heCsN6pHLHKYP45w2TGpehhOjq5G+6EEJ0oOtP7I/Hb/LS0r1Uu/1oGkwbmM3/O+8EbHrzYnnhvPjt3rDHy2q89M5I5OXbT2zPIQvRaUhQI4QQHezWkwZy3bR+7C2pJTPJQV5aQpvuv+tIdcRzOw9HPidEVyfLT0IIEQMJDhvDe6a2OaAB6JkeOYG5VwvnhOjqZKZGCCFiyGeY2DQtpPFkOLsOV7Ng62FMy+KUYTm8tmx/s9skOW2cF2ZLuBDdhQQ1QgjRRpZlhW0W2RbrD1Tw5MKdLN9TWr9Fuwd3zhpCbqqr2W3/9tl2XloamkfTNzORwgo3hqnq02SnuPjtRaPITHYe17iE6MwkqBFCiFZ6Y+UBXlm2j/2ltfTNTOSSiX3w+02+2HoYw7SYOSSHa6f2O2pgseVQJd9/aSVev+r55DNMPlxfyLoD5bx42zSSnIG35sXbDzcLaAAOlNXxk7OHk5bgIDXBztSBWY39oI6VzzApqfaSnugg0Wk7rscSIhYkqBFCiFZ4etEunl68q/H6/rI6fvv+Zhy2QOXeHcXVfLa5iGdumkJWC4HNc1/vaQxogh0oq+ODdYVcMTm/8dj7a8NXIAZYtO0wj1078VheTjMvLNnDS0v3UVrjxWnXOWdUT+49c5hsBxediiQKCyHEUVS5fbzQZBu1x2fgN03qfEZIi4KDZXW8HGZmJdia+saTlgW1XoPyWh9ltV7Kar389bNt/N+8TWytb2BZVuuL+DhNO3Afq+e+3s2jX+ygtMYLqKab760t4GdvrW+Xxxeio0hQI4QQLThS7eHlpfuococGEF4jMNPiN0L7Li1u0qOpqbREB5YF5XVear1+fIaJz7DwGRaFFW7eXVvAzc8u47NNRYzrG7mr9tgWzrWW12/y0tJ9Yc8t3VXCpoLK434OITqKBDVCCBGGZVn8Zf42Lnrsa55cuJOKOh+lNV58RvNlo6Y5w3Zby0nE54/tRZ3PaEzyDQ6JLAvcPgO/afHIp1u5dEIf0hMdzR4jJcHO1VPCVydui4LyuhZnfDYXSlAjOg8JaoQQIoxXl+/nlWX78BkmDpuOTdcwLYtKtx/LApddJdLqmoa9yXbs00e03BH7mqn9SKpPxA0OaDRUgNSQb1Na4+VghZsnb5jEzCE56JqmKhAPyuaJ6ya1S/fwzCRns/EHy06R3VSi8+g0Qc1vfvMbNE0LuYwYMSLWwxJCdFGvLQ+tA5PqsqNpGpZl4fYbuOw6yS47qQn2kO3dI3qlcfWU/KYPF8Jh0xnZO430RAcum66CGQIzPg2PZgHbiqowTYu/XDWeBT89jYU/ncWj10xgeM/Udnmd6UkOTh2eG/ZcToqr1f2ohIgHnSqtfdSoUXz22WeN1+32TjV8IUQnYVkWBeV1IcfsNp3MJAcev8nwHqlcOTmfM0fm8e2uUj7bXIxhmpw0NJfzx/YiwRF+O/ShCjfldV4GZCdz+og8NhVUYk/Q8NX6sIKSjZ12nTqfgdtn8OdPt2JakJ+VyHXT+pPicjC6Txr9s5Pb7fXef84IDlW42RiUP5OV7OThK8Ye9zZxITpSp4oK7HY7PXv2bPXtPR4PHo+n8XplpawNCyGOTtM0BmQns6ekBsuy8PpNPPW5NE6bzhWT87myfjZmzphezDlKFd/Cijr+b95mlu8pBSA1wc7VU/MZ1TuNjQWVJDttVHv8gJrF0TSodvux6xpltT4sS+2CWn9gPZnJarlo9sge/OaCUTjtxx90ZCQ5efbmqSzfU8qWQ1X0SHVx6vDcxiU2ITqLThWCb9++nd69ezNo0CCuu+469u0Ln7Hf4MEHHyQ9Pb3xkp/f8pSwEEI0uO7Eflj1OTRVHj9ev4nXb1LrNfhiS1HYOjPh+AyTO19a1RjQAFS5/Ty9aDezR/bgf84ZwawReZwyNJfpg7KZ2C+TRIcdl92G37QwrUDejQWU16pt159tKuIfC3a062ueMiCLG07sz1mjekpAIzolzQqe84xjH330EdXV1QwfPpzCwkIeeOABDh48yIYNG0hNDb+2HG6mJj8/n4qKCtLS0jpq6EKITuq+19bw7tqCxqUhu66TmmDHpmv85OzhXDn56F+U5m8q4hdvh6/30isjkXd+MKNZy4WT//gFBeVuTNMi3Bt0eqIKelIS7Hzyo1NkiUh0eZWVlaSnpx/187vTLD/NmTOn8eexY8cybdo0+vfvz+uvv86tt94a9j4ulwuXq3kfFSGEaA2PYZKV7MRvmOiahi1ol9Dnm4taFdTsKK6OeK6wvI4ar0FKk6q9PVITOFBWFzagAaio85PktDAtiyq3v8XqxUJ0J502vM/IyGDYsGHs2NG+069CCNHA5zfRoHFLd8g5o3WT3L3SEyKeS090kBgmqfjqqUcPltw+gxqP0eJ2bCG6m04b1FRXV7Nz50569Wo5QU8IIY5VS9uZZwzObtVjnDmyB6kJ4SfFL57Qp1mwBHD5pHx6txAMNbDpGh9vPNSqcQjRHXSaoOYnP/kJCxcuZM+ePXzzzTdccskl2Gw2rrnmmlgPTQjRRV06sQ8Dwmyd7puZyBWTWrfxINllZ+6cEwCVIFzrNbAsOGd0T+44ZVDE+/3y/JEkOnTCzcNoQJLTRqJDZ9XeslaNQ4juoNPk1Bw4cIBrrrmGkpIScnNzOemkk/j222/JzQ1fNEoIIY5XaoKDp2+czItL97JgazGmBacNy+X6E/uTntS8dUE4i7Yd5tfvbcSyLFx2HdOySHDqXDEpv8UE3zNO6MHMITms2V9Oea0Pw1LboDQgPcmOw6aWrZrm4wjRnXWa3U/tobXZ00II0R68fpPzH11MeZhO24Nyk3n1jukt3r/Oa/CfJXt48du9FFS4cdo0Ep027HogGHryhklM7JfZ7mMXIp609vO70yw/CSFEZ7N0d0nYgAZg1+EadhRXtXj/RKeN7546mC9+choXjO1FaoIjJKC5acYACWiECCLzlkIIESUeX8sF+txHOd/AYdP5y1XjWbq7lG92HMHlsHHWyB4M7dE+/Z+E6CokqBFCiCiZ1D8Th03HZzQPXrKSnW1qSqlpGicOyubEQa3bdSVEdyRBjRBCHIf31xbw6vJ97C+to19WEtdM7cd5Y1WpicxkJzfN6M+/Fu9udr/vnzZYKgEL0c4kqBFCiGP0r8W7eGrRrsbr24qqeOD9jRyp9nDTjAEA3HHKYPIzk3h9xX4KK9wMzEnm2mn9OHmo7NwUor1JUCOEEMegyu3jP0v2hj337Ne7uXxSX5Lrt1u3ppO3EOL4ydynEEIcg/UHKnD7jLDnar0GGwsqO3hEQggJaoQQ4hgkOpv3bAo5H6ankxAiumT5SQghjsG4vhn0ykiksLyu2bm+mYmM7hO5QFhJtYcnF+7ks83FeP0m0wZl8b1TBzOsfov2vpJadhyuIi81gdF90qP2GoToaiSoEaKzsCwo3weOJEjOgepicCaBS2qVtFnpbti9CGwOGHImpLQ9aVfXNX5zwUjue30tNR5/4/GUBDsPXDgaTQvfPbvW6+e7L6xkX2lt47Gvth9h1d4yHrtuIs9/vYeF2w43nhvRK40/XDqG3hmJbR6jEN2NtEkQIt4Ub4YV/wZ3BQw4CUZeDHu+gq/+ooIavxs0XV3sCTDkdDj9l5CSF+uRxz/Lgi9+C2teVj+DCmxO/R+YeOMxPWRJtYd56wrZV1rLgOwkzhvbm6xkZ8Tbv75iP498sjXsuexkJyU13mbHh+Sl8NJt0yIGSkJ0da39/JaZGiHiyae/hKVPgln/zX/jO7DwYfBWq2OmH3x1oGkqqEnMgm2fwpEdcNN76gO6PfncKrhKymr/x46FjW/B6pdCjxk++PL30HMs9B7f5ofMTnE1bt9ujUhdtU3LYsfhajKTmgdEO4qrWbWvjEn9s9o8PiG6E0kUFiIeGD5Y/Bf45lH1c+NxL1Tsg7pSFdj4agFLzTJYppq1ASjdBds/bb/x+Nzw+W/hienwz1Pgn6fC0n8GZjc6q3X/DX/csmB9hHPtLFJXbdNs+Xd7sNwdjeEI0aVIUCNEa5TvU0tAFQfa/7ELVsPTp8OC3wGWuljmUQKI+sDGDORycGh9+43pw5/A6hfBW5/3UVsCi/+sgq7OrOZw5HO1JR0yhEj1anRdixjwAAzMTo7WkIToMmT5SYiWuCvgo/th1wIVRGgaDDkDzvlD+yTo7v0G/vsd8HvADNfc8CiBjRb0vSTpOHoCle+DA8vBmQoZ+bB9fvjbrXoeptwKzk76AdtzdOTAtMeoDhnCpP6Z3DRjAM9/syfk+IzBOfTOSOStVc3HN6ZPOmP6yi4oIY5GghohItn5Bcz7MVQVqHwSeyKgw/bPgLlw0WPH/tjeWnj3TtjxOXir1DGraVATLqDRQo/bE9R/bU4YeZG6l9dL7apVWB4PiRMmYGspKd40YP6vYMObgZkhXVfLXrYwya6eKvjmMTi4Qv2cP00FORn9WvOqY2/yrep3HrzEB5CYCWOv7rBh3DlrCLNP6MH8TYfw+E2mD85m+qBs/KaFZVnMW1fY2ARz+uBsfn1BxwRcQnR2svtJiHC+fBCW/0vlsjTQdPXhp9nUz7d9Bul9ju3xP/9flbDqrQFfjTpm1S89hWNzBpaaGoIfeyIkpIMjEc59GIaeSfVXX1P80EMY5eVqyE4nWTfeQNZNN4V/3GVPw6JHQo8Z3vrk4OzQmSBQx50poAcVlktIg2tehezBrX75MbXna1j0Ryjeoq73OxFm/QJyh8V2XEFKa7zsPlJDjzQXfTOTYj0cIWJOdj8JcayKN8PK59QsRjDLVMm6rnT1c/neYwtqDB9sfFv9HBwcaFpQTNOw3dgFdic4ksEywO9Vt0vKhsk3Q2ovGHoWJKThPXCAwl/9EnyBPBvL66XkX8/g6NuX1DPOaD6Wta80P6Y7VDDjd6uaOI3j9qpx6U0q5bor4Zu/wwV/a+tvIjYGzIQB76o6P7pd7eyKM1nJzha3hQshwpOgRoimtn2i/tv0wxtU7osLFVik5x/b43trAgm4Npea+bHqAyhNUzMvuh1GXQJn/BqKNsLiR9TMgjMZBp4Cs34Omf1DHrby/fdDAppg5W++FT6oqTrU/JimgStNLUMFc6WosYaz84uWXnF8kro+QnQ5EtQI0VRDgKHb1bKP0bwYGoNmqYTaY5GQrnJQyvepACIhQ+XVNDxPQibMvAem3aGuDzxZXWpKVG5PQvipV19BYcSn9BVGOJczTM1MNWVzwBm/VMttlQWQMxT2fgsrnw3/OHoXqGEjhOj0JKgRoqnBp8PSp9TPrjTwVAYCDpsLBp0Gcx469sfXNJj2XfjkF+q6blOBjWlA1kC48V3V/qCp5JZ3NzkHDIh8rn//8Cem3g7z7mt+PCVPzRQF73JKSI8c1Aw7p8WxxVzZXlj2FOz9Wi2pjThfLd85pPWAEF2J1KkRoqneE2DkhepnTVcBR2IWpPSEq1+GS/8Zcbak1cZcDmc+AKk91XWbA0ZeANe+Fj6gaYW0C85HTwp/38yrrgx/pxHnwezfQHJQ76O+k+HK/zTftt17Aoy/tvljpPdVM0vxqnQ3vHQFrH8DKguhZCd8/Td449bmu6CEEJ2a7H4SIhzTVCX1N76tdvz0nQyTbm6Wx3L8z2NAdZHaUXS8gRJQt349RQ89hG/vPgBsGRlk33EH6Rec3/IdDZ+qSuxMOXry884vYdO7Kmk6fxqMvVLN4sSrD/9HjTec8/8CI86N+hB8hsmKPWX4TZMJ/TJbLLInhGiutZ/fEtQI0QW5t23D8nhJGDEczdHN810ePxHqwvdbYtQlMOcPUX36BVuLefDDLZTVqiXMJKeN7546mGumdpLaPkLEAdnSLUQ3ljAsQs2Vw9tg/etqGSZ3mCo4l9rj6A/oq4MtH8CBFZCYoYKB3OHtOuaosbewNbqlc+1g95EafvH2hsZCegC1XoO/zN9G38xETh6a28K9hRBtJUGNEF1dZQF8+TvYvVj9rNtUJeIdLlj1Alz+DPQaF/n+1Yfh9RtUbkqDlc/BaT+DSd+J9uiP3/DzYMW/I5yL7tLTW6sOhAQ0wV5bvl+CGiHamSQKCxFN7ko4uFLtvomFNS/D3yfA6pdVsUDTp4rqucuh5ohalpn/65YfY/EjoQENqOrHCx5S29Lj3YnfC18teNzVqppwFB0sq4t47kAL54QQx0ZmaoSIBtOExX+CNS+Cz62O5U+BOX+EtN4dM4YjO+Cjn9VvRw8zW2CZKgm6eLMKWrIGNr+NacDWD8M/vmWqJakTv9+uw253Celw7euw6T3Y+5WqzjziPFX7J8r6Zyfz1Y4jYc8NyJb2B6LtLMvi4NZNlBcWkJqdS/7oMejhCoV2UxLUCBENS59QvaOC7V+uthF/Z174asXtbcMb4G9pNsBShQYNb/gCg6CCGn+Ec6CqI3cGjkQYd5W6dKBLJ/bhjVX78fhCg0pNQxKFRZvVlJfx0eN/5sj+wMxvWm4e5971EzJ69IzhyOKHLD8J0d4Mn8pVCad0F+xe2DHjqC5S/z3aBseEdMgeEv6c3Ql9JkW+74CTjm1s3UR+VhKPXD6O3hmBIn+ZSU5+cd5Ipg1quZiiEE19+dxTIQENQOXhYj59spP0XesAMlMjRHurK4u8hRhU8bfBp0d/HHkjVQVk0x8hsNHUf2bc3fLM0Uk/gjduaV6obuDJUc9J6QqmDcrmre/PYFNhJT7DZFTvdJx2+T4p2qbycDEHtmwMe6608CCHdmyj55D46TQfK/IvS4j2lpChmj9GknYMnb2PxZjLVbVf3U5jABPM5oQhs2HKrS0/Tv5UuOoF1R7ClaIec8bdcNE/ojHqLknXNUb3SWdCv0wJaMQxqalo4YtSK853FzJTI0R7szthzBWwIkyfpNQeMPTMjhlHYiZc/RLM/yXsWqRqzWCq7dzJOTD6Mjj1/tY9Vu8Jqj1EZ1VXppYEizdDz9Ew5kpIadt2aq/fpLzOS2aSE4dNAhPRsTJ79cFmd2D4m7f20ICc/Haudt5JSVAjRHUxHN4CKT3ar6DcSfdBbQlsnqd2CQFkDoALH1V9njpK9mDVr6r6sEoaTukJNYchKav7NHNc9SJ8fD/4atX1dcDCh2DOw61KHPYZJk8u2Mnbaw5S7faTlujg8kl9uf3kQdj0MDNgQkRBQnIKI0+ZxfovPm12buDEKaTnSaIwdOI2CX/4wx+YO3cu99xzD3/9619bdR9pkyBCGD747Dew8R2VdwKqCN35f1ZLLO2hfB8UbYSkHNU/SpMPwQ5VsAaev0D1qWoqIQNu/RRyhrb4EL+dt4n31xY0O37VlHx+fFYnqaosugTTMFj27htsWvQFXncddoeDYSeexIwrrsPujG517Fjr0m0Sli9fzj//+U/Gjh0b66GIzmzRI6pzc7DCtfDWHXDTPNDbYYkho5+6iNhY91rkbe2+WtjwFpwWeQmuuNLNh+sLw557a9VBbpk5kMzkrv1hIuKHbrNx4qVXMen8i6kpKyMpLQ1notQ7CtbpFoarq6u57rrrePrpp8nMzIz1cERn5atTPZDCKdkJexZ37HhEdFQdUoUQwzENtUTYgi2HqjDM8JPZPsNkx+EwM0BCRJnD6SKjR08JaMLodEHNnXfeyXnnncfs2bOPeluPx0NlZWXIRQhA5dF4ayOfL9vTYUMRqJ5Uix5WxQk/+YWaMWsPeSMj5zDpdujV8mxv1lFmYY52XgjRsTrV8tOrr77KqlWrWL58eatu/+CDD/LAAw9EeVSiU0rJA2dS5MAmc0CHDqdbK1yr6uB4gmY9NrwJZ/waxl9zfI89/lpY9ZwKmprK7A8jL2rx7qP7pDMkL4Udxc1nZMb0SWdwbgtb94UQHa7TzNTs37+fe+65h5deeomEhIRW3Wfu3LlUVFQ0Xvbv3x/lUYpOw5Gotl2HkzUIBkS/L5Co9/n/hgY0UN8w8/dQV358j53eB656SSWA6/Xf4XS7qrlz/ZvgSj3qQ/z+kjH0ygjdKZaflcT/Xjz6+MYmhGh3nWb30zvvvMMll1yCzRaofGoYBpqmoes6Ho8n5Fw4svtJhDB8qkP1pneDdj+NhfP+DBn5sR1bd1FxEJ5uobrynD/AqEva6bkOQE2J2ubeUnHEMHyGyeLthzlQVkf/7GROGpIj27mF6EBdbvfTGWecwfr160OO3XzzzYwYMYL777//qAGNEM3YHHDO71UbgPauUyNapyGYjHjeaL/nSu97zFv1HTad00f0aL+xCCGiotMENampqYweHTrdm5ycTHZ2drPjQrRJSp66iI6X2V8t95Xuan5Ot8syoBCiTTpNTo0Qoos67f7wO5Sm3qbaSgghRCt1mpya9iA5NULEqcK1qldW8SZI7Qljr4YR54K7ApY/A9s/VUtRg2fBlNvidmZt95EaXl22j61FVWQlOZnQL4Ppg7MZknf0hGQhRGSt/fyWoEYIEZ881fDqNXB4W+jxtN5w3X9VU844snJvGT96bTUen0md16DWZ2BZFklOO2P7pvOrC0Yyoqe87whxLFr7+S3LT0KI+LThjeYBDaiaMyuf6/DhHM1f5m/D4zNx+wxqvH4avi/Wev1sK6ri7pdXU1HXvMOyEKL9SFAjhIhPuxe1cG5hx42jFQor6thWVAVAna/5ji2v36SizscH68L3kRJCtA8JaoQQ8cnWQgsCm6vjxtFGkXpFAew+Ir2ihIgmCWqEEG1XvBU++DHMuxcK10XnOYaf28K5OdF5zmPUKz2R4T1VMnC4onxOu3qr7dOkMrEQon1JUCOEaJv/3gJPTIcV/1a5LU+dBi9doVobtKcR58GQMI1r+06G8de173O1g/vOHEaCw0aCI7QQaLLTjq5pJDptnD+ud4xGJ0T3ILufhBCtt+xf8NFPw5879X447Wft+3ymATs+g60fgWWqLd0jzo/ceTvG9pbU8Ory/Xyy4RCHKt247DoOm05uqosHLhzF5AFZsR6iECGqS0sw/D7ScnugafHb+kO2dIchQY0Qx+mxqVCyPfy5tN5w78aOHU8cK63xsnZ/OckuOxP7ZWC3ycS4iB9H9u9l8cvPUbR7JwDpuT2YftnVDBg/KcYjC0+2dAsh2l9dWeRznqqOG0cnkJXsZNaIPKYOzJKARsSVmvIy3v/LHxoDGoCKw0V88tSjFG7fGsORHb9O0/tJCNEB9nwNm98DbzX0mw4jLw7taJ01CGqPhL9vunQ2F6Iz2PzVAjy1Nc2OW6bJmk8/oNfQ5o1996xdxaZFX1BTXkp23/6MPeNscvoN6IDRto0ENUII5Yv/g1UvBK5v/wxWvwhXvQTJ2erY7F/B8xeB1aQWi6bDaXM7bqxCiFZzV1djczpwOFUphCP79zaeMw0DLAvdphLcC7ZtYceKpfQeNoKktHQAVn7wDsvff6vxPiUHD7Bjxbec8/176Td6bAe+kqOToEYIAQWrQwOaBqW7YcljMPvX6nr/mXDho/DpLwJLUa40OP0XcML5HTdeIcRR7VmzkuXvv8mR/fvQbTYGT5rKjCuuIyUzC9Pvx1Nbo4KaepqmoddU89m/Hke32Rg3ew6jZ53Jyg/fbfbYpmGw5I2XJagRQsShLR9GPrf1w0BQAzD+GnUpWKN2J/WZCHG8a0KI7mjfhnV8+Pif8dbWYhp+ADYs+IzC7Vs55fpbWPbOG1iWGXKf4H1DpmGw+pN5VJeVhgQ+wcoOFVBRfIj0vJ7ReyFtJEGNEAIMb9vP9R4flaEIIY7f0rdfw11dFVI/yvT7ObxvDyvee6tZQNPA8Pkw/X50uwoP9m9a3+LzaFp8JcHH12iEELEx8JQWzp3aceMQQrSLQzu3hS+IaVnsWbe6xfvWVVdRV1WJ4ffj93qx2cPXhcru24+03Lz2GG67kZka0f5Kd8PaV6FsD2QNhHFXqz4+uh1S4usfgKg3aBb0nwF7vwk97kqF6XfFZkxtVVMCm96GqiLIO0FVJLbHb48oIWKhVaXpLAvLMPDUVJPXfyAnnHQaX70WmnPncLo46eobojTKYydBjWhfO7+E934YWLLY/il89WewJ6kPmD6T4Ixfqg8dET90HS75J6x6Xm3p9tRv6Z56uwpM492uhfDe3eD3BI4teRyufB7S+8ZuXELESEbP3hzes+v4HsSySMnOZvSsM8nO78fmxQuoLishu29/Rp82m/S8Hu0y1vYkFYVF+zH88PRpUH1YXTd9gR0ymg6J2WD6AQuGnAF9p8KYyyFJSse3q/L9sPPz+rYCZ0Bm/9Dz1cWqzUBiZmzG1958dfDkyVBdBP46lbyMpl5j3ylw0/uSyCy6PMPvZ+eKb9m9ZiWWaZGYns7qj97H9Psab2OBWpLStFb1arM5HAwYO5GL/+eX0Rt4K0mbhDAkqImyfd/C6zcFrnsqAt+cLROwAYYKcBzJ4EyG5By48j+QPTgWI+56vv47fPtE/e8b9eY1+VY49aeqsN6iP0LxFlVnxpWqCuYNPgOm3wm2Tjpxu+UDePM28LsDr7uRBuOvhQsfU7NRQnRBht/PR4/9iQNbQtuUJKSkUlNRjunzgaahaSoR2O9tYWNAA01DA+wuF6dcdwvjzzo3OoNvpdZ+fnfSdzERlwxfk+v+Jh8y9dsCLRO8VeCrUZVrv/gtXPFcR42y69q1UC25BLMsWP4vSEiDbx5Vf0beKvDWQM1hKN0FuxfB13+B73wEPTrhsmDx5vqAJtz3Mwu2faKaYvafAeteg11fgmaDYefA6MvA7uzwIQvRnrYtWdwsoAFwV1cx9cJLSUxJA02j97ATePN3v6TySHGrHteyLCzLYskbLzNg3EQyesTP1u1I5KuLaD99JoWW1D/alL9lqqBm60ct9xQSrbP+v5HPffukynNyl6uApil3BbxyVdSGFlX+hm+dkSadNZVA/Oo1sPCPsH+5mlX87Dfw1m3Ng3EhOpmdK5dFPHdwyyZGzzqT0afNJqt3H044ZZbqxn3U92f178nv8VBbUc4nT/yldUnGMSZBjWg/ziQ46d7Add3Wuvv53bDnKzWzIyLbtQDe+QG8eDnM/xWU7Aw9X1sS+b61h1UAGZxI26j+jaryIBxY2V6j7TiZ/cAWaZeTppadijbD4W3NT+9bqhKjhYixgm1b+OjxP/PSz+/l3Uf+j+3Ll7T6vpGK44GqTRNszKwzcaWkYnc6sTmc2F0uHAmJ2F31/4YiBDsF27aw4cv5rR5TrMjyk2hfE65Xu01W/Qd2L1ZJnK3xzg8grbfKfzixE+d3RMuSx1W+DKh8mH3fwNJ/BnYojTgPeo6Fg6vC3z8xCyoOEHE2w7IAE6oKozH66Bp6NiQ9qBKgzeBcAQ3sCeq//hb+Hm6fr5ahOoLfo5bCqougx2jInxp63vCrHYN7v1a7BUecryo2iy5t58plfPavxxtnQqpKSyjcsY2ygoNMvejyo96/35jxFGzfEvZc/7ETQq6n5/Wk3+hxHAyzXGWz2fHU1oZ9HNM02fDlfMacftZRxxNLMlMj2t+g0+Dyf8OZ/6tq09CanSca1JXDkn/A57+J6vA6naqiQK6M6YfaUhUsGl714TfvPvjsARVQulKb39+ZBGOvQgU0LfxZ2JyQOyIaryC6UnLhjF+p3VwNf980Xc0UulIgZ6hKSI8kQmXVdlewGp6apf68FjwEr90AL1+tlv4AvLXw+g0w715Y/wasfgleuQa+fLBjxidiwjJNvn3r1bBLO2s+/YCa8qMvzY88+TSyejcvXZCe15NRp81udvyMm79L3oBBIcey+/RlxEmn0tIXn6qSw0cdS6zJ12ERHT43YKqETI62rKSFFknb8JaarUnr1brn8nvAV9t1tig3tevL+m3KqCWk4Dcd06/OrXlZFTm84jlY+AeVNwLqW/6p/wO5J8Cyp9WOtHDr4pqmqgrnDIn2q4mOcVep17riOdi7WAV+yTkw/Dw1k7XqP3BkR/j7Dmn+pt/u/B54587mS4QFq+Hz38J5j8CKZ8LPtK18TpVAaDqrI7qE0sKDVJUcCXvONAwObN7I8OkntfgYzsQkLvrJL1j3+SfsXrUcy7IYMH4S42bPISE5pdntk9IzuPRnv+HQjm2UFRWSnptH72EncGjndjYu+gK/293sPrrNRnZ+/2bH440ENaL91ZSob5wlO4FWfAtOSFPfrBuYfvUGn9oL8kZCvxPDr/PWlcPCh2DLPJUsmj1YVb8dEduth9Fjtdyjacdnamv2VS+qxGvLCq0BNPOHasbH71YzPVZ9oKTpcMJFcNFj0R1+tOUMhXN+F/7cpJtg28fN85D6TISRF0V/bDs+j5zztO1jNdO0qYXcns3vS1DTRdkd4VsQNLAd5XwDV1IyUy64lCkXXNrq5+45ZBg9hwwLXB88lP5jJrBr1TIwQ9+7Ha6EmG/rbg0JakT7W/xI4MOj6ayA5qj/MK0vAJWYVb9kUM/0qen45c+o5RCAnmPg0qdCP6BNE964BYqC1oVLdqqpe02D4XOi8tJiYtAs0H+rfjdN6fZAQnZwYBhu1mrmPWrGYO3LKgi0TOg1Bs75Y+edoWmthHS45lVY85Kqeq3bYNjZMPbqjmmlUNPCtL1RX6SypfyzcDvWRJeQnteTnPz+HNm/t9k5h8tFv9FjO3Q85931Y778z9Ns/moBhteLpuskZ2Qy86obGDxpWoeO5VhI8T3RvkwT/j5eLT81BChW08z8+mqvybmBZRUArPpvs/XBTvDszLCz4cL6RNnizWpKfu0r9R9ITVLDcoepKrKdScUBlUdRsR+yBqtKy8F9sr59Er76i9qS3Thbo0FChvpdAtz8YeuKGNaVQ+lO9fvP6Ne+r0OEd3AlvHJt+HPJOXDHAvjk55Fna875fcclM4sOV7RrBx/8/WG87kBgq+k6p91w21GXnqLF8Ps4sn8fmqaR3bcfNnts50Ck+J6IDctQH5re6voEzAgF0Uw/uCvBkaiCH5sz8GHtSmu+3LTjM7W75cvfwdaP1eP7alWPooS00C29h7epb72OxCi9yHa2ayG8d1dQvRVgxb/hsqehd/3OhRO/Bz1GqUJ6O+arX6sjMTDLNeXW1ldlTsxQNYVEx+kzCfKnBHKdgk2+RQWm076nZpE8VaHnc4er3CDRZfUYNIQrf/0gmxZ9QcmBfaRm53DCSbPI7psfszHZ7A56DOx8ld5lpka0r71L4IWL1JS6ZRG5IBr1O1Qc6sM5MQP6z4Ttn0UuCjX+WpUQC/UBTdCbf3JOfVIyasfLnctaXycnlgwfPHUa1IRJFMwerGZfmqoshNUvqiTTxEwYcxkMPj3qQxXHyV0JCx6EzfNUAJ+cC1NuUUFNgyPbVd7Tnq/AkaC2dJ/4fbV8JkQ3Jr2fwpCgpgO8/T3Y/omarQm7VVajMdDRdHVJqt9u22ssFK4L/7iJGWo2p7pYPa6nUiW8NrAnqKUYgAnXqcTLeFW6C755TLUnMLwqoHEmh+bENLjxXcjrhNusRWSearUsm5IXWDoUQrSotZ/fUqdGtK/S3aA7g2qGHEXwB/mh9dA3wrLI5FvUNl3LCsorCZrR8XvUklPfyXDyj4/nFURX2V5Vm2TLB2qmyVOpCsPVlYUPAn3hC2GJTsyVAul9JKARIgokqBHtK3OA+q/uUIGNZiMQfAQFIQ1LTPaEwDHLglN/ppJkG5oMJmXDKT+Bad+FnqNVMGP6A4+h6TTWuUnrpbYzO5Oj9/qO1/J/BYqtgfo9oalcJH+T2hCJGarqrBBCRJHXXce2b79iw5fzKTm4P9bDOS6SKCza18QbVI8iUAFNYoaakTB8gSDENGgsYW8PSuZNylIVbc/+Hcz6uVrCCp6in/ZdVUG3Kd2ucg7qylXAkJgRzVd4fPY16eei6arir7dGBWyOpMC5mfdIB+l45q5UfaPK90HWIDjhgvgOqIUIY/ealXz53D/xBhXcGzL5RE6/+bvotk6Ql9iEBDWifQ04Cc58ABY9ooIZ3QE5w2HaHZA9RNWSWf0fKNnVfHlqym2BD3FncvMPiMGnw4QbVGXchm3iNld9Z3At/H3iTbgdWY5kFQC6UtTP2UPUctvQDqh0K47NwdXw3xtVfozNrgL1T34BA09VxR9HnK8SfYWIYzXlZXz2r8cxmjS93LHiWzJ79WHSeR1QmLKddZqg5oknnuCJJ55gz549AIwaNYpf/epXzJnThYqsdRXjrlZVWg+uVIFLn0mB2Zb+M1Qfoq//CuteV4FPeh8V0IyPUMcj2GlzVcM/d0X9ElbQCuqoi+M/T+GEC+HwI82P2xPgosdV3ywR3wpWw/PnBYrlNey10DTY8r6aTVz1PFz5QsuzhqW7VF0ad4XKBRt6Vvz//RVdytYli0MDGguM+p2rGxbM75RBTafZ/fT+++9js9kYOnQolmXx/PPP8/DDD7N69WpGjRrVqseQ3U9xxvCrRFhXauRt3OHsXQLv/1BN/zcYcBJc+Khayolnfi+8fYd6DcHGXa1muER8qyuDJ0+GyoP1B6ygqtlaaJXsiTfC6b8I/zifPaC6rGOqXX32BMg7Aa54Pr6XT0WX8tWr/2HDgs8AMP1+PLU1WA3tETSNqRddzswrr0fTY59+2y22dGdlZfHwww9z6623hj3v8XjweDyN1ysrK8nPz5egpivw1cH2+epDptc46D0+1iNqPdNQTSp3LVAfaMPPVd/URfxb8W/47H9VY1BoXotJ0+urPDtVkvsPvmn+GB/+Dyx/OvSYblfBzLhrVHd7ITrAlq8XsuCFZ8CyqK2sDNmBqdvsJKSmMvWiK5g454IYjlLp0lu6DcPg1VdfpaamhunTp0e83YMPPkh6enrjJT8/dtUZRTtzJMLIC1Wjws4U0IAqCjhkNpz1f6qejgQ0nUfp7qOUKtAC5/1hejkdWBkoIBnM9IO3VjWu7LzfM0UnM2TKdFKzc/B7vc1KStgTVJX2DQvmx2Jox6xTBTXr168nJSUFl8vF9773Pd5++21GjhwZ8fZz586loqKi8bJ/f+feqiaEiLHMASootUdoweFICtReGhCmZ8/WD4hYZdvvVoGN2bRXmhDRYXc6ueDeuaTl5DYe03QdZ1IydofatFFbUY7fF6aZbpzqVEHN8OHDWbNmDUuXLuX73/8+N910E5s2bYp4e5fLRVpaWshFCCGO2ahLVA6YM0Vdgltx2BNUoq/pV7vwpt/V/P4+d31tojAsS83a2TrN/g3RBaTl5DL1kitJTEsnITWNhNQ07M5AKYm07Fzsjs6TwN6pghqn08mQIUOYNGkSDz74IOPGjeNvf/tbrIclhOgukrJUo9HMfmpWJikH0vpCer6aYfFUquT39HzV/mLvElU/qcHAk1W+TbjAxpEAM+7usJciRIMhk08kJTMb3WZDa7JpY8zss2M0qmPTqb8SmKYZkggsREz5vWr3S7xty/V7YeuHsO9btTvshAsC3b9F2/WeALfOh4JVqk7N5vfVJSlbnTf9sO8beOFitRPK7oQJN6r2HUPODHTr9tWoJSfLVAHSBY9CvxNj+tJE92R3Ojn/3vv54tl/cnjvbgAcLhfjZs9hzKyzYjy6tuk0u5/mzp3LnDlz6NevH1VVVbz88ss89NBDfPLJJ5x55pmtegzZ0i2iYuPb8MX/QcV+1fdqyGw467eQEQeJ6e5K+O9NUNRkmXbq7ar9hDg+deVqi7fhVdctC+pKA8UhEzMDszIDT4bKAlWB2J6oAmC7U3Wnn3qH6souRIyVFhzEU1NNdn4/nAkRcsdioLWf351mpqa4uJgbb7yRwsJC0tPTGTt2bJsCGiGiYulT8PHPaEz+NHyw+V04tBZu/0J9qMXSt080D2hAVWUeMrvz7RyLN5UFgYAGwPAEAhpQsza6A7zVKvhNzFbBjFGfeDn7160rOilEB8nq3SfWQzgunSaoeeaZZ2I9BCFC+T3w5e8Iu5ulfJ+qmDztux0+rBCb31PjtEy1LBa8HXnLPAlqjldab5Uj0xDYNN25pNvV776x27pFSGPXbx6F0ZdLjy8h2kmnShQWIq7s+1Y1ogzHMmHH5x07nqYOrIAj21ShOG+VWhZxVwTqoPjC1FERbZOYoeolNQjeDaXb1SxNQ8DT0FE+WG0pHN4S7VEK0W1IUCPEsbKMQE2ScFwxbK7pqYK3vxf6IQtqecRXH4j1n9nx4+qKTv+lSr7WdNVgVdPVrFhChjrfsJukIY+mKVdKhw1ViK6u0yw/CRF38qdBQhrUljQ/p+kw+baOH1ODLR+qwMaRrHY/BS+R+epUQDM0ivlo7krY+YVa+uo/Iz6SpqPFkQDnPQIn3wdHtqvX/PVfVUd6CBTk08K83fYYBVmDojc2dyWU7oTkPNU4VoguToIaIY7F3iWqD5BdlRLHskK/hY+5HAZ00EzIroUqd8ZbA/lTVY5GVaE6p9tVsrKvpn4ZRFNF4i78e/S2nm96F+b/OrC8pekw/ho1o9GWxqWdTVpvdQEYdpbq5l1bCj3HqCWmd+9SAU+DpCw458HojMU0YNEjsPZlVfBP01Qge86DkJIXnecUIg50mi3d7UG2dIt2seVD+ODHgV4phlcFFI5EyBkKM34YmmcRTZ//L6x+KfRYRj+YfCt89uvw90nvC7d9Fp0Ao2QnPH9B+FL/Z/4vjLuq/Z+zs6gqgg1vqg7fOUNVdeKE9Og811d/gW+fbH48bwRc8hS4y9UMUbzVVBIigi63pVuIuGCasOjh0OZvNickOtWsyFUvQWqPjhlLwermAQ2onVdFG9SHVumu5uen3Ba9GZP1/43cu2jdq907qEntAdN/EP3n8XvC/72wDJXc/o/pKphJylL1cSbfHP0xCdFBJFFYiLYo3alqk4Rj+mHvVx03lq0fRT63/RO44lkYeEoggEnKgtPuV0tB0VJ1qIVzRdF7XhFQdUjlU4WwVKFAwwtmfY2c2lJY8AdVekCILkJmaoRoC9tR6olE6t4cDUYLnXMNL6T2VH2KqovVVu6M/tGvh5I3MnKwlXdCdJ9bKMk5KtcrOH/HH1QUUGuyI27Z0zD2yo4bnxBRJDM1QrRFZn/oOTr8OWcyDDoVSner6rG7F0VeimkPg05t4dyswM8peSqHoyMKvI25XM0INaXpatlLRJ8zWeXrBDP96r+arXlgXr4PvLUI0RVIUCNEW81+QG3lDqbb4Yxfwae/hGfnwEc/gzdvh6dPh8K10RnHwFPDBzYJabHr9pyUBVc8D30mBY5l9IPz/wL9p8dmTN3RaT9TO7AaaDZ1SUhvnk+VmKF2xAnRBcjuJyGORfVhWPcaHNkKqb1gzJVqdmb5v5rfNjET7vhS7Y5qb4YP1ryktlF7qlWX5ym3qRmlWKs6pLpQZ/Tv2lu541nJTijerGZvPvhx+ArY076rauwIEcda+/ktQY0Q7cHwqV0lzRI0653zexh9WceOSYhg+5fB+z8KLRY54lyY80fZ2i3inmzpFqIjeasjBzQQeceUEB0lfyp8d6Eq1uguh94TIHtwrEclRLuSoEaI9uBKVwm51cXqumWpHUhG/Q4Ub61KGm7aiykayvermjCle1Q+y7irIHNA9J9XxD+bA4bOjvUohIgaCWqEaA+6DpNvUXU/LAs8lYGARrPBimdUIbyLHovuVP/eJfDO91Rp/AarX4QL/waDT4/e8wohRBxo8+6nm266iUWLFkVjLEJ0bpNvhpn3qKTYhoDG5lS7S9Bg1wJVJj9aTBM+/X+hAQ2oGaNPf9VyXRshhOgC2hzUVFRUMHv2bIYOHcrvf/97Dh48GI1xCdE5Tf+B6kqdmAmJWWqrrKdaFb/zu1XjyWg5tBYqDoQ/V3MYDiyP3nMLIUQcaHNQ884773Dw4EG+//3v89prrzFgwADmzJnDG2+8gc8n3wSFwPCCZleJww3LUIZH/XxwZfRmTPzeo5z3tHxeCCE6uWMqvpebm8t9993H2rVrWbp0KUOGDOGGG26gd+/e3HvvvWzfvr29xylE59F/JvjrAj12gnmqVU2ZaOg1LnLXZ2cS9J3c8v0NH+z5GnZ+0fJOrs7IU63qCK36DxRvifVoWmZZcGg97F/efClRYJkWRpUX0xvFat2i0zquROHCwkLmz5/P/PnzsdlsnHvuuaxfv56RI0fyxz/+kXvvvbe9xilE5zH+Wlj4UPPjmg6OJNj6oWon0N4cCaqI2vxfNz83/S5wpUa+747PVTXkhhomziSY8cOu0cF5x+fw4U9DC88NOxvO+1P81Wc5sBI+mQtle9X1hDSVpzXh+sBtDm+DpU/CgWXgSoORF8Ok73RMG4wYq1lVRPU3BRgVXrCBq38ayVN64hqQjmaXAvniGIrv+Xw+3nvvPZ599lk+/fRTxo4dy2233ca1117bWBDn7bff5pZbbqGsrCwqgz5WUnxPdJjnL4SDK9Q3bctUxzQNdAfkT4HvfBC95961AFY+r3ZbZfaHCTfA0DMj375kJ/znovDLYhf/A4acEbWhRl11MfzrjPBLc9PvhJk/7PgxRVJdDP8+J3zV3wsfVW0PijbCa9c379U04CS47F9dunJzzaoiKj7aA4DlN7E8hprVsmnYsxNJO70fSeNyYztIETVRK77Xq1cvTNPkmmuuYdmyZYwfP77ZbWbNmkVGRkZbH1qIrmPIbChYXR/Q1H9vsCyVW7P3G1j4MJz60+g896DT1KW11r4aOc9n9QudO6jZ9G7kXKN1r8VXULPudRXQNHzPDA5QVj6r6iC9/V2oLFQzTPYENfsHsOcr2LMYBp7S8ePuAJZpUf1NQePPltsfOGlYmFVeyj/YhS3diWtAhCVY0S20eb7uL3/5CwUFBTz++ONhAxqAjIwMdu/efbxjE6LzGnIm+OpoDGiCWaZanirb1+HDCqtif+Rz5XEyxmPVUAwxnJojaht8vNi3BOrKoPYw1B5RO+Ya8rIOroKXr4Ij21Rg7K2GutJA922A3YtjM+4OYNb41JITYPma59JYhgUW1Cw71NFDE3GmzUHNDTfcQEKCdHQVokUHloIzJfJ50wcf/rjjxtOSrEEtn9u9GP57M7x4OXz+W7Ws1Vn0GBX5XN4JqmhiPChYrWZaTJ+aqbEMlWxeW6KCr7qGfk1BszeWGbpU1YVzajSXDc1e/9rDxaG6Ouc/UtdxgxJxKU7+RQvRxRi+wNJAJLsXwqJHOmY8LRl7FdhdYU5YULpb5dtsegd2fg5LHoXnLlDLHZ3B8HNVq4hwpn2vY8fSkm+fVIUagWaze5Zftdiw/M3/nAxPYLlqxAVRH2as6E4biSOz6680OanRGPDYMuULd3cnQY0Q0TDwFNDthHyzbsrmhOXPRC6Y11Ey+8MlT0J6n8CxxAzoMVrNIAR/yBo+qCmGz34TX0s3kdidcOV/YPCsQJCZ3gfOeRCGnxPbsQU7uFK107BFmG2xLPDWqdk/rWn/MAumfRfyRkR9mLGUdmZ/nPmpaI6g16+BnhD4d5Y8uUdsBifihvR+EiIacofDmMvULiRfmN0smg72RLWEsPNLmHhDx48xWP8ZcOtncGidClyyBsA/ZhA2J8j0qyWo4o3Qc0xHj7Tt0nqpoK2uTNWrSesTP8tODRIzVG0gy0R9QDf9vVtqOcqVCklZqjq14YWUHnDZM0evQdQF6Al2cm4ciWdvJbWrinBvKcMy1e9Ld+qknppPwtDMWA9TxJgENUJEy1m/UwXxPv8/qAxKxtXtqkhew8xBwy4Xvxe2fqBmRxIzYdQlHdtdW9eh93j1c9Gmlisfm/7QJNV4ZBpqh9PGd1Q15z6TYMpt8RfQgPqz/vrv9bMwTXdr1Qc5lqkCGZtTBcSudLj06W4R0ARz9U/D1T8Ny2/i2VsJpoWzXxq6q+kMluiOJKgRIlp0HcZdrS4vXl6/xFBfq6bxNna1U6rmCLx+E5TsCJxb+hTM/g2Muyr84xdvUb2kPJWQPw2GndN+xeTSeqsCfJ5Kws7WJOeq5al4Yprqg99mh6pD8MrVqq6LpqvtzyW7YNvHcPXLaiYtnky5HQrWwI5P1YxMI039nbFQhRuTstQSVO/x6j494+zPoANpdp2EwRmxHka7MKq81K4pxl/qxp6VQNL4PGypx5b4bflN6jaW4N1fhZ5kJ3FMDo7cpHYecfxqc/G9zkyK74mYObQe/vsdtfwR7KR74cTvwUf3qxmFpnQb3P4FpPYMPb70KVj8p9BjPUbCFc+rKrTt4eOfqzo13iZj1h1w+TNwQpwkplYVweJHVE0abzUkZYO7CtzlobVe7AmqAu/gWWo5Kh7t+QpevCzQp6th/M4UFdSc9rOuUeVZNPLsraT09a1Y3kCOmu7UybxqOK5+bfu3bFR7KfnPJvyl7sDfHQ3Sz+pP8uSeLd85zrX28zsO52GF6IJ6joEb34XJt6jlguFz4IrnVEBj+GDLh+HvZxqw+f3QY0e2Nw9oQC0ZffP34xtn0UbVLuGNW9VMzbBz1FIH1M+COOGUH8dHQGOasOoFeHwqLP+Xqknjq1O1ddxlqCWboO9sDXkouxfFb5LzgJNg0s1qJiwhDRyJKhgzfGqn06DTYz1C0Y4s06J83q6QgAbA9JqUv78Ly2z9nIP3QBXFj63Bu78Ks8aHWesDwwQLKj/di7+8ezS0leUnITpKel847f7mxw2f+rCNpGnZ/M3vRb7tpvfg9P93bOPb8BZ88otAW4c9X6kZA90BLqeaNdLtsPzfkDkIxl5xbM/TXj75Oax5qb75ZlDV5hAWITvQDG/9DqI4bidw6v9A8SZVcM8XtBRld8Hbd8BVL0KKtAPo7Pylbtw7yzHK3GH/PhrlHnwHq3Hmt9CzLeixSl7ejFERFLiYFmatH81lwzQsSv6zkeTJPUmamFe/Y6xrkpkaIWLNmdR8F5HpB1+tuuQ22aobrjdQa861xFMNX/w2ENCA+tlTAf5a9YGqB70Rfv23lhOJo61wnWorEKlqc7Cmgc6ws+M7qEnKgrMfVLlBjiRwJqslNXsClO2Br/8a6xGK42BUeSl5aTPFT6yl4sPdmDU+LI+fcH+PW9uJvGb5Icwmsz2Ncb7HAMPEX+Km8sv9HHl2I0ZNDP/tRpkENULEg5n3qJkQy1LJuXWlKj/E8MKHP4E1rwRu22965Mfpd+KxPf+exc2bJDbMHoXb6VRzWJXsb2/uSljxb3jruzDvXrXdvSlfHbx7p1piMtv45pzaW+Uxxbvtn4DNVZ9Lkxxam2ZLFJuhiqiyLIvS17bi2VOpDthUcG35TCxPaFCiu2w4+x59lgbAV1iDpmmNlZXDVQTQ6p/LX+qm+uuDx/wa4l2nCWoefPBBpkyZQmpqKnl5eVx88cVs3bo11sMSon0MPFnVG0nvpZZ8NF19S09IVzMinz+gdjsBDD4dek9o/hh2J8y469ieP9z27KPNZjjaeUdF9WGVJLvgIdVpfMuH8Pb3YP6vQ2+36BEoa+gt14oZF01TuUD9Z8Jtn4YWGYy2qiL44nfwzFnw3Pmw5PHWzab5W8h/MLxhltk6P8uy8OyuoHpJAbXrDrd6lqIz8e6txFcU+PKgaRqaUwWsqqdV4M81ZWbvVm9Tb9gppTltEScuTX8gaHJvLWvjyDuPTrOwtnDhQu68806mTJmC3+/n5z//OWeddRabNm0iOTk51sMT4vj1n65qjySHyZewLNjwhsqX0W1q99GSf8Dmd1VOSf6JMP0Hqi7OMT33TPXBH5zbY3PWBzZ68yq2PUZB1sBje65IljwWvoHm2ldh5EXQZ6IK8Da+rZbDvDWB7c7h3sk1PVAT6IK/q+TsaC87GT7Y8TmU71W1hr55NLSp5pHtsGshXPVChNYU9QaeDCufC39uwEnxvXx2DMw6P6WvbcV7MLDTTv9sL5mXD2vzDqB45j/ibnZMc9pA07B8BprDhiM3keSpPUkcldPqx02akEvdltLATE04PhPLYaHpGrQhAbmz6TRBzccffxxy/bnnniMvL4+VK1dyyimnxGhUQrSz2pLI52qOBH52JsOpP1WX1ijbCyuegX1LwZWigoRx1wTq2iRlwfQ74au/BN1Jg4QMVFAT9GaZmAln/66VL6gNtn4U+Nmy6gMsSwVXWz9UQY23pj6Yqc838dXWBzZBb9INdWmc9cs2U26FEee2/3ibKt0Fb94GFfVT+54q9RoSM0KDwsK1akfbmMsjP1b/mSp4adpjy5EIM37Y7kOPtYpP9oQENABmnUHZG9vpcfcENEenWVRokS0zfCCrOXT0JDs97pmA7mr7x7JrUAYpp/Sh8qM9Ld/Qb4LTRsKwrlt5udMENU1VVFQAkJWVFfE2Ho8HjycwjVtZWRn1cQlxXHqOCZ9HAsdeaK1kpypE5w76+39oA+z9Bi5+IhCwnPg9yB6sdhRVFkDOMLUFPaWHmiWqLIDsIerDODEKb4oN+TF+d+iOJjQ4sFz9mJCudpFVHFD5Jjanur1pqCU0V6oKaBqk9oRJ32n/sYbz/j2BgAbql4kM9Xtv+vvataDloEbT1J/NyudU/R1PBfSdqno8xVvhwOPgN/18tWsxPdf60S0dp82JUw8UnTPr/Li3lpI4uvWzFvHMNTAde3YC/pLmMzZJY3KOKaBp4MhLQk+0YzYkATfttlG/EdCW5iRlZgcuwXawThnUmKbJj370I2bOnMno0ZHf6B988EEeeOCBDhyZEMdp6u2we3HzHJeUXBhzjFuov/l7aEDTYOeXaiZg4MmBY0PPVJemZt5zbM/dFoNOU1vSPU3HaqnZje3z1dim3gHzf6VO2ZyBJpDZg2HIbNj2ifr9DZ6lbpuSF/2xF66DwxESp02fGk/w7rFIjSuD2Z0w7Q516YK8hpe5i+ey98BOfu2/HQCP4calu0hxptCQL9WVdupoukbWlcMpe3s7vkMqt0bTIOGELNJm9z+uxzbK3Gr5ya6BUR/NBAc2Nkg+sRepJ/XBlnJs1Yo7g04Z1Nx5551s2LCBr776qsXbzZ07l/vuu6/xemVlJfn5+dEenhDHrs8kuOQJlQx7eKtaShlwEpz+CzVLcSwizfwA7PoyNKiJpRl3h6+qbHOqnUCrX1RBzbirVKCw9EmVXKzbYcgZcMavITkbTr6v+WNEW+2R5sfsCYFmplaT7bbD50R/THHunR3vsLp4NbpDp9JeQ5pf5UZ6TA9Ow4nTppZqnL1TYjnMdmfPSiD31jF4C6oxqrw48pKwZyYc/Y5He9xsVSRTd9kx63whE51okHHBEFJO7HXczxPvOl1Qc9dddzFv3jwWLVpE3759W7yty+XC5WohGU+IeDTwFHWpPqySSY+37YHNGXk3TXAfqljLGgQDZsKOL9TSjaapwMBRX9G44kDgthOuh7FXQ1WBCvaONeBrLz1GqwDUW62ShTVd/dkZ9uazNMPnwOAzYjfWOPHF/i8AMDWT+bnLuKxwVuM5j+HFaXPhGpDWquJzndHxBmuWYeHZU4FZ68eZn4JrcEbj0pae5MDymWrGRoekyT2OOaBxby+jblMJlsfAnptI0sQe2JIdWKaF7oy/JqKdJqixLIu7776bt99+mwULFjBwYDvvvBAi3rRX1dhhZ8P6N8KfG35O+zxHe+k5DvYvD38uZ2jodZsdMvpFf0ytYRoqfya4T5a/TtWYGTJb5dTYnCqgGXJmfHYK72DeoJ12i3JWYWkms4unkeFPwa/7SRqfS9rsOPnzjTPeA1WUvbUdo0otzWkaJI7JIfOKYZS+sgV/Ua2qTePUSZrSk/SzB7T5Odw7yyl/dyf+w/Vb0HUNnDpVCw6oxG1Nw9knhdTT+uIaEOMvFUE6TVBz55138vLLL/Puu++SmprKoUOHAEhPTycxMTHGoxMijs28B/YvhfL9occnXBe+3k0sjbtaJSoHtwcANfPRUQm/x2LhQ+q/jqRAlWOtvsP2nIfiJ/iKI1N6TGFPxZ7G64uz1/BV1lpS/UncMOFGRo0cFLvBxTHTo7a/m+5AHR/Lgtp1R/Duq8Ko8IJTFfK0dA3fwWosr4HWhtYIvkM1lL2+FaMqqMSDYUGdgaWpTuB6kh3Pngo8/y4nYUQWKSf2xjUo9sFNp+nSrUWoy/Dss8/yne98p1WPIV26RbflroT1/1XBjTNZbekedFqsRxXe/mUqEbi0vsBeSi6c8j8w8sLYjisSvxcenRjUNqK+kaZWn8xwyk9UArgIcaTuCN//7PuU1IWWMeiX1o/Hz3icZEf4+mOm20/t6mK8eyvRXDYSx+SQMKTrblFuqmZlERUf72l23PKbWB4DLcne7PMy9aTepJ7a+nzSsnd3ULvmMJY7aMNCpEhBA3QNPclB8pQepJ81oNXP0xat/fzuNDM1nST2EiI+JaSpei1Tbo31SI4ufyrc8rFKlPZ7IG+kWmqKV6avSR8sLbSuj7/59l0BOYk5PHb6Y7y85WWWFCzBptk4Jf8Urh1xbcSAxqj0cuSFTRhBHafrNpWSPLnHMS2xdEZGZYT8OL8ZsdJ03abSNgU1/qCqx0dlAYaFWeuj+ttCEkdmt7q9QzTE8TuFEKJb6yz1WJzJ0Hs8FKwJf37ASR05mk6lR3IP7p10L/dOal0/rqqF+0MCmgY1K4pIHJPT5XZKhePIbaE9iaaFX9VoYwVhPdWJVlwbmJxpzd0NC6vWT/WyQ2TFMKiRbDUhYqjK7eOtVQd4etEuvt5xBLMLly/vNCxLFSws29P6+5x0X6A6c7ChZ8Zf3lInZJkW7h1l1KwqxvIaYT+k3ZtLYzCyjpcwIgtbRvCu3vrlTpuG5gz/ke4aktGm50iamKcCpGPY3VS7qgjPvtgVupWZGiFiZOmuEn721npqPIF16+E9U3n0mglkJDmxLCtiLpmIkp1fwoI/BAKa3GFwxm+g76SW79dvmurntOxpKFitdjuNugQm3RztEXd5ls+k9HXV2bohx8PyGmhOW8iHrukzsMz63kZdmGbXyb5mBOXv7cSzq1xt3dZAT3NiS3I07ohqYEtzkjKjd5ueI3F4Fv6T+1D9dQGmrmF5/GAS6B/b0ncv06L8rR3k3T2hsTN4R+o0icLtQRKFRbyo8fi54LGvqHb7m507oVcaPsNkR3E1uakuLp3Yl5um98duk4nVqDq0Hl65pkl+DKpOzo3vQubxVXwVx6Zq4QGqvlLtJ0y3X+WO1NOTVL2UhiDHluwgcUwOqaflt7rDdWd15D+b8OytVLM0ulp2siyLxGGZGJVeLL9JwpAMkqf1auzi3VZGlRfPznIsv0nl5/uw/Jb6fdf5VJATiaZq42ReOrTdvph1uURhIbqSL7cWhw1o6nwGa7bv5eKktUyxqthUMYR/LvRwoKyWX18wKgYj7Qa8tWrpaMWzzQMaUFu017wMs+Z2/NgEtesPN/6sOW1Yhtn4gWrW+euXXnSwaZgeg5oVRfiLa8m+YWSMRhx9nj0VePdX1c9KBYIGTdPwHa4j7/vj2mXGypbqJGm8ajOipzopf3uHOuGyY9U1f/8KVrehBFd+GslTex73ONpCghohYqCstvmHp2VZTPSt5gHnv0kOKky2yRzCb9bdw03TB5CX5iLJKf9s28W+pfDVn1WCr82hAhrLVDVxmjq8pcOHJxQrqB6LpoMV/FndkFtjCy374dlXhWdPRVwVhWtPvoKaiOf8ZW68+6tw9EhCb0NtmqNJHJ6F867x1K0/glnjw1fqpm5NcfMZm4Y/Bl2jZvkhCWqE6A7G9Gn+ZptoVvO/9mdIwEvwt6+R1g6u977O1U8nYZoWvTMSuW5aP66YfAx9zMr2wIEVqpv1oNNUKf/uxPBDwSoo2gSL/qgqAYMKaNzl6npSFsG/fwBSu37PnHjl7JeKe3s5AJbHDOR2BPc28ppgV8swDbz7q7tsUKMnN09KtywLvCaWz+DwU+sAsOckknnJkHb7PdhSnKRMD+TnVOYmUfn5XvBbof9kHDqaruEv92AZVofm1khQI0QMjM/PYMqALJbvCezYON1aTgLeZmvQlgVnsJR/eK7B5nBSUF7Hw59sxe0zuGH6gNY9oWnAp/8PNr4dqGWRmAHn/enYtxxXFkDJDkjrozpkx7tdC+DTX0J1sQpgDB84UwK9pewJ4K6AmsOApmZvnMmqvcG4q2I48O4t5aQ+eHZXqHyOoHwadEISVi2/GZI4rCd13Y+3hBFZ6PP3YnqCqgp7DRXcQeMMlr+4liPPbCD7xpEkDG3/AoVps/Ixyt3UrChSz6lpqoWCQ8122jJcHZ4sLJmHQsTIw1eM5fJJfUmqfyMekOzBYdObzhEABsnUkWoPXbJ6fsle3D6j2a0xDSjbC3XlgWPLnoYNb4UW56orh3fvhJqSpo/QMl8dfPBjePoMePN2ePZceP1GFSzEE8sCb436b9keePfuwBgNH2CBtwoMj2o66atVxyxTXQyv+h1N+o5sy44hZ+8Usq87AdfAtMDfX7uOluQAe9BHmGWBaWH5DLAsEoZ13SrDustG5qVDG5OhLcsCX31A0+QNxPKbVHyyJ2pjST93IPacRLRkB3qyQ+1Kq/9iljKlY5eeQHY/ie7AXQGaDVzxWZjLZ5jUeg3SCr/BfPN2Kup8GKaFDYMk6nBYfgzNxiptFM/oV7BWP6Hxvv+5dSojegb9XV77Knz7D6gqAt0Gg2bB7F/DS5erY+Gc+lOYclvrB/zxXBUgNdVrHFz3eusfJ1pME5b/C1b/R3U6T8mDlJ5QuDZQ6be2RDWgtCzC70/V1O/PlQaDZ8Hl/+7IVyAiOPLiJrx7qxqvW5alElbN+qWn+hkKLcGOPd1JxqVDcfXruu/1psePe3MpnoNV1C49hOUNvyVJS7TT454J2DMSWvW4lmXh3VuJv8SNPTsBZ/+0Fncx+Q7VUPbODvwlqnq2ZtdIntqT1NPyZfeTaGeGH6qLVJl8V+yqPMbE/mWw6JH6DzMdBp4Cp/0MsuKrw7vDppOeqMZn6z2OzMJ1eHw+HJ5KNCwsDdy4GGzt4wHjb/xYm8tOTW0vTk8MWlvf8BbM/3XgumnAjs9UD6WqQzT7Cteg6lDrB1tbCpvfD3+ucK2q0RLrWY2FD8HK5wLXq4vhyDbQ7Wq5CdRSk7eaiAU3NE3N1niqYO+SaI9YtFLqyX0pPbAFy6gPXjQNEu3qj8tjqk80h46maRg1fsr+u428uyZ02e3dustO0vg8nAPSqF0a+d+xZVlorSwJYVR4KH19K77iQFNZR14iWVcOx5YePgfP0TOZvO+Nw3uwGtPtx9krGT0pTDHKDiDLT13ZimfhqdPg6dPhH9Pho/tVY8PuoGgjvHGr+qAF9QG1awG8doP6YI5HmgaX/Qtt9KUkaAY2zULTbdRqyXhQdSYc+Lnc/BiAKQOy6JUe1KF+6T/DP27pTkjOjfy8uSNaP8bKg+G3PTdoSxXeaKgtVV2+m9LtatnMqv8m62ih1DwEbmeFWd4TMePqn0bWNSNw9U9Vk2kuG8mTeqA7bWguW8jSB4DpNnBvauPyaidkz0hoMYiwJTtbXaum7J0dIQENgK+4jrKG7dwtcPZJIWFwRswCGpCZmq5r5XOqMmoDwwcb34Hy/XDNy7EaVcdZ9rTKiWiq5rDqVj3tux0/ptZISIdzHlSza7sXo2k6CaaJp87XmE4w1NpL38xEfnFeYBkKb03LAUXeSNi9qPnxtF4w4rzWjy+tT2D7cziZA1r/WMfKNGDXl3BogwrWTjhf/d4ADq0LPzZHogpqTL9K/NW00B00Lckd1p6jF8fJ1T8NV/+RjRW3Tbef2lWR87mMKi9GlZe6TSVYXhPXwLSYNlxsb6bXoPrrAoyaMO93ABrYs1u37OQrrsV7oDrsOe/BanzFtTjyjvKFIMYkqOmKDL/KKQjn4ErYvxzyp3TsmDpa4ZoWzq3tsGEcs5QejfVSHLpOVrITj8/EtCx65vTltVun4wieTrYnqA/u2lLAAptL5YQ0GHSqyg1Z8o/63T2obthn/V9g909rJGXBCRdEyKkZG/2lp5oSeOM7cHhb4NjiR+CCv8PAk1V7gnA0mwp8knMCs5WaLWgmJkx00xBFFqyGJ2bCuGtg2vfiu2N4HPDsKqduUymYFq7B6SSMyGr10kdbNMzIaC4btgxXs0aXlql6IvnL3BQ/tkZdB6oWQcKwTDIvHRKVcXUky7QofWWLCkQsDTQr8FdZB+w6utOGo1f4rudNGVURAqOG85VeCWpEDFQXqQTJSA6t6/pBTVIOVBZGOJfVsWM5FmOuCAkcdDQSHSpISZ55g6qgGmzFv1Xyq7fhW1a1WmJxpoAzCUacr7Zwj7kCSnepBNi0Y6y9csav1KzHtk8CyzR9J8P5fzm2x2uLL34bGtCAqgg871747kKVrJw9RG01b6rnaLj+bRWkGF7Y8gEsfUKdsxr/r179EoZuB1sC1ByBbx6FqkI4+3dReGGdn2VZVHywm9q1gfee2vVHcOYXk33NcDRHdPJaNE0j5cReVHy8p3EclsdobKdQs6T+fcCmEog1Xce9rYzqbwtJndknKmPqKO7tZY0zK5pDV9WWG2gausuOpmskT+rRqsdz5CWpAodh8o01XeXWxLvOHaaK8BLSw3cMbpDSur/gndqYy8MctNT23bQ+8Z9b1GeiSmrWg753aBpMuF41Sgy2d4lKiHYkqhmaBr5a9U504aMqoAH19yJ3+LEHNKCe54K/wm2fwaX/hO/Mg6tfUruMosldCdvnhz/nqYLtn6qfz/9z8xyilFw4909qliV/CgyYqWapsgY1fyxNUxebQ/1bCt69seEtqDjQPq+ni/HsKA8JaBp491dRvawNyejHIHlSD9LO7Ict2a4qEDfUswmOUw2wavyYbj+WZVHXZKxGtZeqxQcofXUL5e/vxLu/injn3RP0PqZroXsBDLWzL+Piwa2eXbGlOkkcnRP2XOKoHGxp8V+sU2ZquiJXisqT2PhO83OJmTBkdocPKazD22DLPPXh238GDDwN9HaKs8dcqZaZGmY7/G41i+FIgq/+CkufhBk/hCm3ts/zHS93BZTvU1uPU+o/kCffrP4ct88H06cqAIfLWVn3Wv0PmvoQNv1qJkLTod+Jx15c72jS+6hLR/FUqdcWSV2Z+m/ucLhtPmz9UOUZZQ6A4ec2X2bTdbhjoco92/gm+NzQeyL0GAWrXwz/xcAy4eAqSO/bXq+qy6hrISHXvbEk6rMiKVN74eybwpFnNmL5NVWMLhyfiQWYtYG/S/4jdZS8uAmjJnCsdt0R0k7PD6mgG2+0+iJ3jVvbLQK5YvUBjlnTQmJ/GOlzBqI5bdSuPYzlM9EcOknjckk7o1+7jj1aJKjpqmb9QiUFH1wZOJaYCRc/Do7WJY1F1ZJ/wNd/C1xf9YLK8bj06fYZn66rhNuJN8H612H5M5CQGcgz8blh4R/V9u7Bpx//8x0rwwdf/l4ldnur1YdmQjpk9FfVbPvPgMm3QGoLRaxqmnw71u2BGZ7aI1Ebept4qlTeT0sziEeT2lPNMEVaVgzO53EkwujLjv6YrhQ4+//UpUH5vqBAMYyGWS8RwvKFWbNoOOePfK49+UvcIfVqIt/QxN4jMHtR8ekeVdLfX79V3K4q41Yt2E/iyOyIW5ljLXF0DtVLClWbguCSc1pDwKNRvfQQyZNbXwRPs+uknz2A1NP6YlT5sKU60F2dJ1ToPCMVbZOQpnY5HVihcmhSeqgZmnjo9XNofWhA02D/Mlj2FMz8Yfs9V94I9Y/dHiFQWv1ibIOaL3+vZo38qmgVlqlyY2pLVBB2ZDts+RCufTXy7EDPMerPOZweY45/jFs/hlXPq1yczAEw4Qa146g1ts+Hb/6uZuUcCSq359T/CexWagvdBifeqdo9NDXgpPZLUs7oB30mhX4haJCSC/1mtM/zdDGugem4t5ZFPNcRbKkuGtecWoprLLD8BqbXT936I9RtKAkJCixDBWJ6op26LaWkTIvP3l+OvCRST+1L5ad7Q0/oWmPLCKPcg+k10J1ty2nSXfZOFcw06HwjFm3Td7K6xJNN77Z8rj2DGoCK/ZHPlbdwrq0Ob1OJp/uWqlmWkReqSr2RdhfVlcPaV4ICmibvwp4qFYTWHIZvn4icoDr+OrVN3dNkK6bdpUr8H4+Vz6vAq3HMa1RX68oCmHZHy/fd/hm8d3fgdfncsP4N1fH62v8e21Lj2CtUcLP0n2ppyZUCoy6Fk+9r+2O15JwH4b83hc4KuVLh/L/J7qcIEsfkULuqqFmNEz3JTvL04wsKLNOibsMR1SHaY+Dqn0by1J6NtVeMKi+VX+zDvbk0ZFkpIg08e6s49PBKTI8/dGanIS/FsNTskxHfRfdTT+qD5TOo+vIAauejjmbXaHghtmQ7mr37pM/Kv07R8TwtJOB5opDAmz0Edi+OfK49FG+BV69RO3FAzbR88xjs+xau/E/o9uoGZXtUfZlILEMtT2k67Pwy8u0y8uHyZ1XwUbBaHesxCk6be3w1Vnx1asdPON/+A8ZdrWYEI1nyWPNADVR9md0LQmfITBN2L1SziknZgd1a4Yy+VF08VWBPjE6QkdkfbvkEtn6kZsvSequt7C293m5Od9rIvmEk1V8XULe5BAwL1+AMUmb2bnV5/kjK39tJ3cZAzo6vsIa69YfJvnEUeoqDkhc24S9TW7q1BDuWu4XARkPtHvQYmD6zeaHtoHwUy2/iGpJxXGPvCCkzelO7uhizrnkeUdLEHmh60xfZdUlQIzpe/tTwScwA+dPa//nGXQNrXgZ/aB0LNP34ZzJAleF/7y61lV631y911b+JHFihApKhQcnZPrdKkN6zWCX0WlboDptGFrjrp/P9tSowG3hy+DH0GquWqKoPqyWs1HbY4Va4NnIA6qtTyzODZ0U474bizZEf++CqQFBTVw5v3qqCnQaLHlZbxFtaGox22w+7C0ZdHN3n6GL0BDtpZ/Rr16RSz97KkICmgVHjp2rRATSnjregpr5LNOrSMPNiQwUpDSk9DQGNDhgctfiiLd0V93VZQC0VZV05nLK3d2BUqlozmgaJY3NJ6eTb1ttKghrR8Yafp1o4HNkeetzughO/3/z2lQWw4U2oOAjZg1UCaFtqzWT2h0v+CfN/pZJAQeVGnPJT6HecQdTGd1RAE/zhr9eopGytfnZmz+JAUFNTAq/fACU7gx7EOuqbKwDvfB+ueqHl3JGU3Mjn2ipSHlKDlloN2JxqCS7STFRwkbwFD4YGNKCConn3wXcXyexIN+feFj5PB6B23WHVnbqhPktwPnL9LiDNYcMyLJU4q2touqaWnADNpoGuYZlG6L9BXSUKp5zceQICZ99U8u4cj2d3BZbbj6NPynHPkHVGEtSIjudIUEsy3/xdNUf01aldPjPuVssmwXYtVEGDP6jS5fJ/qa7JTW/bkv7T4dZPoXiTmh3pMfr4duKAmhV5647m7RhMv9qinVgfeAXn1Cx+JDSgSUiDWkNt2W78mln/7lpfURjNBo5ktRS17Gm4+B/HN+7W6jVOJc2W7QV/ncr9sUzQHWrJq6VcLV1Xsxyrw/RhsjnUUg6oP/stH4Z/DF8dbPsYxl553C9FdD2WZWG5jfr8ESI0W1fBieX3g00LVCHWNbXK5NCxdE3VtWmY3dE19CQHjl7JJE/sXDW9NF0jYXBGrIcRUxLUiOYOrYfDW1UeQf6J7Vc7JlhSFsz+jbpE4vfCxz8LDWhALVd8PBdueq9tz6lpKhAyfGqWyJmsZnGOhWnAK1epYn7hGD4VAGh6oLeSaaoqtiF0lUNieCA9X+0uOrIt0MfJ5qzf4l7/Z3Bo3bGN91hoGpz9e3jxktAZF8ML7nKVv9NSYHPSfWoJ6uCqwDGbA+b8MVCoz1MdvkdXg7rI39JF95AwPJOaJsX7LMOsr0NjAXr4gMZS25PRNPQEO46eyfiKa8ECR69kjEovltdUC8WJdlW/xm+qYGZSD5LqG2WKzkWCGhHgrlC7VfYtDRzLGqhmBsJVXo22vV9H7qh9eCsc2QE5bUz0Xf0iLHk88Li9x6sP7uzBbXucr/96lADDUkHNtDvUlmtQib9N83oa2FxwwoVw6k/h8/8NP8MBqv1DR7I7VTIuqBkozR4Ishb/ueXmqK4UuPpl2POVyr9JzFAJwMlBryE5R80GNSwLNhXtXlIi7rn6pZE0NofadarmkuU1QgrrWT4jfHNSm4bm1Bt/zrpiWP2sjYUtzYm/pI6Kj3bj2VuFpmnYchJJPaUvSePacQlXdDgJakTA/F+FBjQApbvhnR/Aje+BrwZc6dGZuQnHV3eU8y3sHGrK8MMnc1WPJIv6AnWa6lz94qXw3cWtz93w1qrEY80ORKjWqTvUB3pwzo7NoWY2ItWUGTBT/Xf05erxw+0cGnNF68bYXnYvqm/6EqYh3sGVaqbFlRL5/pqmkpsjJThrGky/Ez66v/m5/CkqqVx0e+nnD8I1KIPKhfvx7a8KJPwGF/Wza2r7df3kjZ4UWF52DUxvVkDPnp1I9vUjMSq9mF4De1ZCt9ol1FV1n83romU1R8L31bEstRz16AR4/ER4+jRVu6Qj9J0SOe8lKQvyRrbucUwTnpihCvuZfrD8YLhV0OR3q1mCl69SS0qtUVmgPswdiTTfD4o6Nvbq8EnIM38U/jX1nwH9pqufe4yE03/Z/HajL1M7uTpSuGThhmBLtx9/XhKo3JtzHw7MBjqTYcJ1KrlbCFTTSs/eCnwHqgO7mRoqGDfE/oaFlugAm1puauDITSTj/MgzzbY0J46cRAlougiZqeksdnymWglUHFBLJZNuVsmv7aW6OPyHuqdS5XvoNrUMUVWklkdWvQDeKvUBNO4atTW6PT7ggqXkqjYHy//V/NyMH7b++eb9CI5sDX+uIe+lcC1sfq95s8hwknMCz924wydoViVvFFwQoWN1/hS1g+nbJ9SMTWKGes6pd4Ru655wHQw9SzVp9Lth4CmQM7QVL7adDTtHLTOZhpoZ89UBlgpoBp7SfhWqR16oLtGsPSM6LffOMqq/Ohi6uwnUPzubhmbTwbJImdGLpHG5+A/XYVR4sOcm4hqUIQFLN6JZVrg57q6psrKS9PR0KioqSEvrRNtEl/8LFj4cekzTVNXT1nwIt4anCp48OXTJx/RDXX3uSWKW+iAzPCpRFyuwOweg13iVuBuN2iFrX4M1L6oZkuwhMPlWGHZW6+//u94tL1VpOjhTYNjZcFmYACqcD38Km+oTlU1ffa6MpZJ9v/d11/pQXva0ak0QnNCr6ervxEWPt+3PQohjUPzkWrx7KyOWPtCSHSQMySD7mhEdOzDRYVr7+S3LT/HOXRm+qqtlwaJH1C6b9uBKbb600fDYNmd9g0RLjSfcO8uhdfDN4+0zlqbGXQU3vQ93r4RrX2v7h6j/KLk5aGqZ5Wg5PMHO+FUgB0Z3qKAobxRc9VLXCmhAJTo7k+tnUFzqtSZlqcBmSYSKw0K0I39J/b/NCBMumkMn9RTpnC5k+Sk2DqxQ23bT+sCAk0MTbysOqNLsvlrof5KaKfG5wz9OzREo2tB+O0RO/rGaAVrzsvqAtznU8kLD7EvDNuVwLFN1w571s/YZS3tyJKulskgS0tUH9MBTWv+YrlRVK6doo+pllNJT5cR0VBJ1RzqwQgVurjDLfYe3qV1zx9KgUohW0hPtmNX1X7Ka7HTSXDZyvzMKR88wyeyi25GgpiPVlMDb34UDy1WOgqarPInLnlZJkiufgwUPqWl+vxusP0BaX7UVWItQL8HmPLaxlO1VSybZQwIfxDa76qB84g9UcJWYCS9drvJtGrWwWumrPbaxRNuE61WjyXAS0tXvMKOf6mXUVj1Gtb4IoLsSVjyjglbDB4NOU7k0afHZAbhRS7vCbEFbvkHtMtu1AMp2q9/p4NPbP9dKdDtJ4/KoWnQgsJU7aMYm46IhEtCIRhLUdKT37oadXwRmOzQNCtfAf2+Gi5+AT3+lcj/MhmZsGpRsV0FHQmb9ElCQjH6t3wHU4NAGtXW7aKO6ntZbtQsYcW7gNq4UyKtfm77wURWI1ZXXfzjpgBmaT9PwWgae2raxdJSzf6fq2uxeEDrTZE9QSymjLoWTfhTd2QZfnWqPULwlcGzNy7BjPlz3Zvv0aoqW4eepnK5wRfJGnKtq2YDaRfbGraE1Z9J6qzylttYBEiJIykm98e6rxLOvSnXOtiywaaSc2IvkiXmxHp6II5Io3FGKNqltxeFmOjQdMgeqb7fhlnd0m5r+T8gIHHMkwMVPtm0HVHUxPHtu8yaFmq6WUiI9lrcWtn0ElYWq/9KKZ+rL+gdJyoYb321b64KOVrQZFvxOzSTYEwOBWWIGXPF8IJCLhtUvqV1j4Uy6CWb9PHrP3R62fKBqyQTncOUOU7+3hj5cL12pdpE1lTtM5UQJcRwsw8S9pQzP7go0p07iqBycfVqokSS6lNZ+fneqmZpFixbx8MMPs3LlSgoLC3n77be5+OKLYz2s1vnsN0RcurFMKN0Z/hyobyXOFFVxtvIg1B5RdVLeu1uV1R97peqlE9xjKJx1r4fvumyZKlCJFNQ4k1SNlAZDZ8Nnv4aSXWqGpv8MmP2Aqq8SzzLy1QxV00JydeVq9uq616P33HsWRz63e1H8BzUjzoM+k9W299pSVYl58BmBpOjD20IDGsuq70BuqoC+cK3qJSXEMdJsOomjskkclR3roYg41qmCmpqaGsaNG8ctt9zCpZdeGuvhtGznQlj7sgpCMvrD/qVHv08kml0tPQ06Feb/UiUIe6vVudKdsPcb1fX6qhdCS9A3FalWC6hk19Yacoa6dDa7FqhZp3AK16o8ovQo7aBoKffpWPOiOlpqD5h6e/hztSWBnw0feCpCZx0XPARXPi/5NUKIqOpUQc2cOXOYM2dOrIfRsurD8O+zoHRX0MEWvqU3Cte8pJ4jQSWTrn1VJZs2BDSWpe7jq4F9S+D5C9VsQ6QP5tQWElJTe7dijJ2cP8IussbzEfoytYfh58K2T8Kfa2h42ZnlDlfBmd+tml02/bt8YDl8+w+YeU8sRtellLnL+HjPx+yt3Euv5F7MGTiHvCTJKxECunidGo/HQ2VlZcglqvZ8A38+oUlA0wo2l/oG2zT5FtSuJ0eS2pF0YHngg7choGlkQfFGePGKyE0gx1zZPNm4wfgOLr8fC/1nhP8dA6T3UXlN0TL0rNBk7Aa9J8CEG6L3vO3NXQnLn1H9wD6eG+gVlpSl6gk1FCFsYFkqH8yyYM0rqmWFOGabSzZz40c38vS6p/l0z6c8v/F5bvzoRpYfWh7roQkRF7p0UPPggw+Snp7eeMnPz4/ek9WVq+3PTRNom9EI2Y+o21UCsCutvnFgomoaaXOqQKf/TLXle8T59R/IVpiAJkj5Xlj7SugxwwcFa1SOwzl/UIXUgp9/6m3tV5k4nqX1hok3Nj+u6XDyT6JbY0bX4dw/qY7nJ1wAw8+BOX+AK/+jcpbilbtSdTb/4v9UccXnL4CFf4Qdn8OGt+D1G1UbBYDT5qo2EJoqWd/YI8r0QV0JlO2BmsMxeyldwR+X/5GaJtWxvYaXh5Y9hO+o7z1CdH2davmprebOnct9993XeL2ysrL9A5tdC2D9f2H/8tZ1jdZt4EhVDRU1vb7DsabyZtLzYeINqpdQel8YeXFgZwmo4nDbPwVaeB7DA/uXqc7HABvfVpWHa46o65kD4Py/qOfwu1XQFM/bidvbrLmqNtDaV6DqkFo2mXKbmsWJNl3vXPlIh9bDm7fVt8Wgvg+YVwXhwTN+S/+pgu7cYaoH2KH1qiBf0y3ghlc1Dp15t1qOO1piuwixs3wneyv3hj1X6i5lTfEapvSc0sGjEiK+dOmgxuVy4XK1U8O9cL7+Gyz5h/q5rqyVg0qDO5eqGinfPgFb3lc1TPrPVIFI7vDI9z31f1RCq7tCFeRrpn4GqKFY2t5v1BJB8K79sj3w/j3wnXnRS4qNd2MuVxcRmWXBBz8OBDSWFchJ8lRAYpMdKNs+UkHN0LMh5Y/NZ2QsE9Dg0Fr48H41u3PZvyDvhGi/ki6j7ijtPo52XojuoEsHNVFVcQC+fTJwXbdBuDgjhKaSQlPqk/pOu19dWit7sGoa+fFc2PhWk5o2mtperTtg5EXq0MrnQgOaBr46NVMx/W41k7PrSzVrNPQstTQiO1REwWpVdToc01BLSnrQ3xN//ayM3Qkz7ob37grUtLEsGv9+giouWXUIXrxMVVXOGaaqOaf2bN/XYJqw9QPY/L6amew3HcZfGzr72Unsr9rPoZpDOG1OvGGKIDp0B+NyZcu8EJ0qqKmurmbHjh2N13fv3s2aNWvIysqiX79+HTuYHZ+HBhXOlKO3Ceg5Fs778/E9b0oeXPKk2mGy9+smO3Y0GHsVDJmtrpbsCPcISvFWVeG2cF3g2M4vVZG1S56UwKa78zRJqtc0lefV8IHaNFgeFFRNus8kVQHbMuq3dzd5LNOvZnsAtn2s/i2tfkHN3LRXHzPLgg9/ov4+NziwAja8Cde82imWXOv8dczfM59Xtr7C/sr9OGwOPH4PbsNNqiMVmx5onXLViKtId0n/LSE6VaLwihUrmDBhAhMmqDe+++67jwkTJvCrX/0qBqNp8qau6SrBN5y0PjD7N3D7F4GS8sfD5oArnlMNKLOHqB5NvcarpNML/xa4XUvLS+7S0ICmwZ6vVIE10b31Gtf876ojmcZE9+BO5ENmQ/7UwPXswaqDuWZrvttMd4YG/w1fDDzVqgBie9n7TWhA06CyAJY81n7PEyVbS7dy7QfX8n/f/h+bjmyiyltFuacch81Boj0RE5MkexKDMwbz0yk/5ZbRt8R6yELEBWmTcKzK98EzZzX/xmoakJAKfadA36kw9orYdTDe9qmqOtyUzaGSkiNtPR90Glz6z6gOTXQCi/8ES58KPWb6VdBi+NTf61GXwPjrQoMcUDlm8+6DvV+pRq5Y9Tv6XKEd05NyAoGP6YeBJ8Oer9XPfSfDnIcha0Dbx/7Zb9QW8nASM+HOb9v+mB3EMA2u//B6CmsKKfOUhXx/stvspDvV+8mfTvsTE/LaaWZLiDjXJdskxJWMfqq6atM3/YRUuPxZVUY+1oadBaf8BL55NLBMlZihWhpE6loNrdiWLrqFk38MyXmw6j8qiM/op/pUTbj+6PdNzIQrnoWSnSo42vaJ2jEVXAAxuP+WaahAaPP7avbGstROv51fwCVPw5g2VhAP10OtNefiwOri1RTVFmFaZrMJYb/hx7AMbJqNguoCCWqEaEKCmuNx8o/VNP36N9Ruj55jYOJNkBXFIm5tNfV21Rtq39L6ujczwO5SbRGKI7RGGHx6x45RxK+JN6iLaR5bHZ/swXDR46q32MrnVYKwZlNVsoN7cPlq64MZCPkkN/3w5s1QsRdOurf1zzv4dFj7Wvhzcb6lvqI+30jX9LCFxi3LAg0GpA3o8LEJEe8kqDleQ2YHEnPjVUK6mrUJNvFG2Pph8x0uPUbBqDjvqyU63vEUJtQ0VQto0i2qltPyf6uWCcEad/REWA3/7DdQWwZn/LJ1SewDT1X/Lnd8Fno8JTdQwylOnZB9AoZl4Pa70dAwUTNLGhqapmHTbYzMHsmonFEYpsE3Bd+wqngVCbYEzux/JoMyBsX4FQgRO5JT053VlsKq59UUv6bDsHNUyX5XSqxHFn+O7ICy3WoJpqVaQqJ1Ns+DNS9D5QHIGa46eRdvJGJQA2r5aujZcPkzrSvcZ/hV6YOGLd39p6uZ1IaSCnHq/Z3v89slv8VjeLDq/wcqqEl2JHNK31O4f+r9JNoTuX/R/Ww4siHk/t8Z9R1uHBWmcrYQnVhrP78lqBGiJe4KmHevSl5tkD8Fzv8bJGdHvp9omxXPw7x7aDGoQVOJxbN+pmZ+uqBSdylXz7san+Gjzl+H23BjWRYaGv3S+vHPM/9JrxTVmPaZ9c/w0uaXwj7OE7OfYHiWBN+i62jt53en2tItRIf76P7QgAZUS4x5P4rJcLqsiderPJuWaPUF/LZ82DFjioHFBxbjN/1omkaSI4lMVyaZCepS46shOahv2/y98yM+TkvnhOjKJKgRIpLy/aq3Vzj7l8HhrR06nC6ttgQcR1n2tDnVMmmYirpdhcfwhFzXNA1d09E0DQsrpJpw08aWwVo6J0RXJkGNEJFUHAjfZqLx/P6OG0tXd2SbajXijBDYaEHnBp4a/jZdwNSeUyOeG5Q+iJzEnMbr4/PGh72dhYXf9PO3VX/j1S2vUuZuZV86IboA2f0kRCSZA9TMQKS6JplxtHW/s0upb1vgTFElB7w19bWV6vtGJaSrROG0XqoTeBc1IH0AcwbO4aPdH4Uct+k2bh97O+sOr+PzfZ/j9rsZkjGEZYXL8AXVlWrYNfXZ3s/Q6nttvbDpBX4787dM7DGxQ1+LELEgQY0QkaT1gmFnw9aPmp8bdKqqwSLaR85Q6DMRDq5SjTITMlQw6atVwUz2EBg8SyUIp+TGerRR9ePJP2Zo5lA+3PUhZZ4yhmUO45oR17DowCLe2PZGyG37pvalZ1JP1hxeQ6I9EZ/ha9z63aDOX8fvl/6eV85/BYcuPd1E1yZBjRAtOft3Kjl12yeq6q2mq8Ju5zwY65F1Pef9Gd6+Aw5vU9c1XTW4vOSfkN4ntmPrQLqmc/GQi7l4yMWNx9YdXtcsoAE4UHWAk/uczB9P/SP7q/Zz00c3oWs6HsNDnb+usfqw23CzsmglJ/Y6sQNfiRAdT4IaIVriTIbz/wJVRVC2BzLyIa13rEfVNaX1ghvfg33fqppAmQOg33QVVHZzn+/7POK5L/Z9we1jb2dV0SoqvZV4DS8Wahs4qCWpGm8N83bOk6BGdHkS1AjRGqk91EVEl6apInn9p8d6JO1iT8Ue5u2aR3FtMdN6TWNqz6n8bdXfWHxwMXX+Onok9eDGkTdy+bDLQ5aMmnIH98xqes5ws/zQch5d/Sh+099YrC+4aB8aLD+0nGpvNSmRkrGF6AIkqBFCiCh4bsNzPLr6Ubym2ob9zo530DQN0zLR6zee7qvcx8MrHqawppAfTvxhxMea3HNyxNozk3pM4pkNz2BaJk6bkzp/Xch5C4tEWyI+08fm0s1M6TmlnV6hEPFHghohhGhnK4tW8ueVf26cLQEVXIQr4O7xe3hj2xvU+GpYWrgUt+FmUo9JfGfUdxiaORSA0/JP4+3tb7OldAsWFh7Dg9fw4tAd9Eruxed7P0fTNBJsCbj97pDndegOkuxJACQ5kqL8yoWILalTI4QQ7ezBpQ+GBBZNNTSpbFDhrWDernmNOTFLCpZwz5f3sKt8F6ACk0dOfYTLh11Oja+Gam9141LTC5teoNpXjWVZ2HU7Nt2GHvQ/p82Jpmn0TunNyKyRUX3dQsSaBDVCCNGO3H43eyv3tvr2DTM4DYm9wY/z4uYXG6+7bC6+LfwWj9+DhoZlWdT56qjyVmFZVmM14mRHMhYWJmZjQ8xEeyJzp85tMW9HiK5AghohhGhHftOPXWv9yr6FhaZpYWvIrCpa1fjzgv0L2HhkY/jnq5+hqfZVU+mpDCQJaxoaGoZlsK9qX0ihPiG6IglqhBCiHaU4U5jQY0KLtwmelXHoDlIdqWFnUZIdgQaWX+z/IuKSls/0keJIwa7ZVa+o+v9ZlkWtv5bC6kLmLp7LNfOu4YNdHxzjKxMi/klQI4QQ7eyOsXeQ6cwMey7NmUb/tP70S+vHBYMv4O+n/x2nzRn2tmf2P7PxZ4/hiXg7E5MKbwVOmzOkRo2FhWmZjUtchTWF/GnFn1h0YNFxvkIh4pPsfhJCiHY2Pm88/zzrn/xt1d9YUbQCv+nHqTuZ2msqP5r4I4ZkDgm5/fUjr+fFTS+GHBubO5arR1zdeH1qz6msKFqB1/BimEbIbTNdmViWCmBayE9u3H312tbXOKXvKcf5KoWIP5oVbo9hF1VZWUl6ejoVFRWkpaXFejhCiG6ioLoAp80Z0mW7qW1l2/hi3xe4/W4m95zM9F7Tsem2xvM1vhru+vwudlfsxm248Rqq/k1WQha/nfFbfvH1L7AsizJPmQpwgnZYNSxHpbvSset2nDYnH1/2cfResBDtrLWf3zJTI4QQUdY75eitNYZlDmNY5rCI55Mdyfx11l95dcurLDq4CMM0mNF7BteecC05iTmMyx3H2sNrSbQnUuurVTk1QVWFHTYHdl295fdIkurYomuSmRohhOgCjtQd4f99/f/YVroNt99Nrb8WwzLQNZ0EWwJJ9qTGZOQ7x9/JZcMui/GIhWg9makRQohuJCcxhydnP8mGIxs4UHWAXsm9KKot4rHVj1HtqwbUFu8LB1/IpUMvjfFohYgOmakRQoguzGN4WFq4lFp/LeNzx9MzuWeshyREm8lMjRBCCFw2l+x0Et2G1KkRQgghRJcgQY0QQgghugQJaoQQQgjRJUhQI4QQQoguQYIaIYQQQnQJEtQIIYQQokuQoEYIIYQQXYIENUIIIYToEjpdUPP4448zYMAAEhISmDZtGsuWLYv1kIQQQggRBzpVReHXXnuN++67jyeffJJp06bx17/+lbPPPputW7eSl5cX6+GJGHHX+Nj0dQEHt5ah23QGjsth+LSe2Oyxjdkrj9SxbsEBDu2swOGyMXhCLiec1BubrdN9lxBCiE6hU/V+mjZtGlOmTOGxxx4DwDRN8vPzufvuu/nZz37W7PYejwePx9N4vbKykvz8fOn91IXUVXn54B/rqCp1hxzvNTiDM28dGbMAoryolg/+sQ6v2x9yvM+wTM68eSSarsVkXEII0Rm1tvdTp/nK6PV6WblyJbNnz248pus6s2fPZsmSJWHv8+CDD5Kent54yc/P76jhig6yfuHBZgENQOHOcvasOxKDESmr5+9rFtAAHNxWxoGtZTEYkRBCdH2dJqg5cuQIhmHQo0ePkOM9evTg0KFDYe8zd+5cKioqGi/79+/viKGKDrR/U0nEc/s2lXbgSEId2BL5uSWoEUKI6OhUOTVt5XK5cLlcsR6GiKrIyzixXODRbTr4zLDnbDZZehJCiGjoNDM1OTk52Gw2ioqKQo4XFRXRs2fPGI1KxFr/0dkh103DwjSssOc60sBxORHPDRgb+ZwQQohj12mCGqfTyaRJk/j8888bj5mmyeeff8706dNjODIRS6NP7UNGXhKm38Rd7cNToy6Gz8QwYpcDP+HMfqTnJDY7fsKMXuT1lyR1IYSIhk61/HTfffdx0003MXnyZKZOncpf//pXampquPnmm2M9NBEjCckOTrpyCO/+dQ2aBppdx2bX0O0aX72+ndTMBHoM7PggIjHVyfl3j2PHymIKd5TjSLAzeEIufYZldvhYhBCiu+hUQc1VV13F4cOH+dWvfsWhQ4cYP348H3/8cbPkYdG97Fx1GJtdb1aXxrIsNn11sMODGr/PoK7KR1Kqk5EzezNyZu8OfX4hhOiuOlVQA3DXXXdx1113xXoYIo6UF9VFPlcc+Vx7MA2T/VvKqC51k56bROHOcrYtK8Lr9uNMsDN8Wg8mnN1fCu4JIUQH6HRBjRBNpWYnULgzwrmshKg9b3lRLfP/vZHqclXg0VvnxzQtXIkONB28bj/rFx7E6zaYcemQqI1DCCGEIl8fRac34sSe6BEq9I6YHp2dcZZl8cV/NjcGNJZpYfhMLMNqVnRv+4oiaiu9URmHEEKIAAlqRKeX3SeFk68chisxMPFod9qYdsEg+o7IispzHtpZQcWRwNKWaQZ2Wpl+Eyv4umFRVlgTlXEIIYQIkOUn0SUMmpBLv9FZFG6vwLQseg1Ox5kQvb/etVWhMy9ak4kiywot/peY5ozaWIQQQigS1Iguw+6wkT8yOjMzTWX3SQm5rtt0NJuGZahoJng5LDc/laxeyR0yLiGE6M5k+UmIY5CRl9SsYrEz0Y6ma9gdtsZpmvScRE69dngMRiiEEN2PzNQIcYxOuWoYy1N2s2NlMX6fSWKyg/Gz88npk0pVqZv0nET6jMiMmMQshBCifUlQI8QxsjttTL9kCJPPHUhdtZfkNBc2h0x+CiFErEhQI8RxcrhsOFzN+zwJIYToWPK1UgghhBBdggQ1QgghhOgSZPlJCNGhqhYupPzNN/EXH8bRsyfpF15A6umnx3pYQoguQIIaIUSH8BUVcfDHP6Fu1Sp1QNPwbt9O7YoVeHbuJOf222M7QCFEpydBjRCizbx79lD+9jv4DhzAkd+XjEsuwdm/f8TbW4bBwft+TN26dUEHLcy6OnRNo+yll8m4+GLsubkdMHohRFclOTVCiDap/upr9t5yCxVvvUXtsmVUvPkWe2++meqvv454n5pvluDdswcMo9k5s64Oy++n5tulURy1EKI7kKBGCNFqls9H8SOPgC+0Ezk+P8V/+hOW3x/2ft7du1t4UAssC81ua8eRCiG6IwlqhBCtVrd+PUZJSdhzxuEjuDduDHvO3qsnmq6j2cOseGsamstF8owZ7TlUIUQ3JDk1QgQpL6pl8zcFFO6swDIsegxOZ+ysvqRlS3E9IOJMzNHOp5x6KkdynsDy+zEqK9XsTD09IYHcu+7Clp7ermMVQnQ/EtQIUa9geznzn91IXZUP028CULSnki3fFHLiRYMYfWrfGI8w9hLHjkVPScGsrm52Tk9LI2HMmLD3051O+vzxIQr/3y/xHjiA5XZj+f04Bwyg5/8+QNK4cdEeuhCiG5CgRoh63767E0+tvzGgaeB1+1n+wR56Dk4np29qjEYXH/SEBLK/eweH//TnZudyvvdddKcz4n1dQ4fS/5WXqVu9GqO8HNfwETj79onmcIUQ3YwENUIAZYdqqDhch+Ezm5+0wPCbbF9R3O2DGoCMiy/G0as35W++gW//AZz98sm4/HKSpkw56n01XSdp0qQOGKUQojuSoEZ0eztXFrPk3Z3UVnhDjmta6O08Nb4OHFV8S542leRpUzv0OX1FxZS9+gq1y5ejO12kzD6DjMsuQ3e5OnQcQoj4JUGN6NbWfL6PJW/txDKtZucsqz6w0cBm18kbkNbxAxQA+A4dYv/3vh+y88qzfTu1S76lz5//hOZwxHB0Qoh4IVu6Rbfl8xqs/HBvIKDRwt/O4bKRkpXAkEl5HTc4EaL0hRfCbiWvW7OG6sWLYzAiIUQ8kqBGdFvFeyrxugNbkLXG/6u/rkFCioOhk3tw7vfG4EyQic1YqflmSeRzX3/TgSMRQsQzeZcW3ZZuUzF948JTww/1gU2/0dnM+e4YbHaJ/WOtpeUlqUQshGggQY3otnoMSCUhxU5tRZMEYAvQYMT0nhLQxIjl81H+/9u796AorkQN4F/3DDMMjwFFXqOAoAZ8IoYbFt1sEiUPY7haW+u6KZLVkPyzZd2I2fWqyboma9RoKinjJmti1qsJaow3u6YSo0vQNSaWriIGC43Xd4SI+AiPGd4wfe4fCkgQBJQ+TM/3q6LK6YbmO0XJfJw+3b1tG1w5X0KrroZiMkFobihq+wIT8NBDEhISUV/EUkNeSzWpGPYfETi6q/gW+xRoje0XD/eEq6wOJrMKP3vH93DxFK5//QsVW/8XDUVFsERFIXj6rxCYlnZXv4cQApf+uAjV+1tPKwlNg1ZdA9XPD4qptdgEPvww/FJS7ur3JyLPxVJDXq2qrB6+AWY01rnhdl8vMSaTCh9fFeeOXsM9KRE9PvbFk+XI23Ee5aU1AICwGDt+NjUOIQMD7kp2vZVv2YJr7/y15XXdd9+h9JU/o7H0Mvo/ldHj4zZdu4Zr761FzZEjMAcHwy81tU2hAa7f30YNCIBl8GD4OCKhWn0RmDYJ/r/4BZSfXntPRF6LpYa8mrvRDVVVAWhQbiyq0dwa6qs1VF6t6fFxr/3gwq4PvoPmbp3tuXLBiZz3j2Hq3CT4B3nWvVW0mhqUrd9wy31lH36IoGnTYArwv/1x6upQ9sGHcP7zn9CcTlhiB6PmyLcQdXUAgHoA1fv3Q7FYYLK3vYReURQ0XbqEwRuz73Q4RGRQXDBAXs0xrB8a693Q3O3vJOy8Wgvnj7U9Ou6xvRfbFJpm9bVNOPnv0h4dU6a6Eyeg1dy65InaWtQdO3bbYwghUDJ/Aco3boT72jWIhgZUH/g3RE1NmwdcAoCor4fW2P5mh7wfDRF1hqWGvNrw8ZHQbnHjPUVVYPJRcfbI1R4d98eL7R/42JV9fZVi9e10v2rrfD8A1BzKQ+2RIwCuFxzhdgNuN25saPf5orZ9oeSiYCLqDEsNeTWrvxlWXxNMFvX6jWkUBSaLCqufGYqqoKG26fYHuQU/e8enl2weuGDYd8Rw+Dgct9xnDg/v8OncN6s9kg+haXC7XHCXlcFdXt72E5qLzY2fg9Dazp75DByIkGcze5SfiLwD19SQV1MUBeFxQbj8vRO4xWRDeGzPHo0QnxKB0vOV7b8fgPj7er74WBZFVRG+cAEuzp8PUdM6g6LYbAh/cSEUtfXvo9pjx+HcuQOa0wnFPwDuigqoFgsEALfT2To7oyi3nKFp3tdvxgxA06BVV8OWlAT7E1NgCvDMRdZEpA+PKTVLly7FF198gYKCAlgsFlRUVMiORAYx9uFo5K473u401ICBAYge0b9Hx4xLCsWPJVU4/k0JxI03btWk4L4nYhEa3f0nfTeVl0OrroGPI7JNgdCTbexYDN60CZWfb0djcRF8BkXB/sQU+IS1Pj6ibOMm/Pjee9dnZCorgaYbM103Zl+gaa3/bt5+8wzNDeawMITP/2+uoSGiblGE6OhPpb5l8eLFCA4Oxg8//IB169b1qNQ4nU4EBQWhsrISdjsfTkitSk5X4NvcIly54ISPxYQh48Jw72MxsNjurPe7yurww/+Vw2RWED0iBL4B3XuTbrx8GVfeeAM1Bw8BmgZzRARCnpkF++OP31Guu821Zw/KNm5CzcGDrUWl6Sen7m4uMD8tZpoG+PhAUVXYkpLgeH0lfEJD9QlPRH1eV9+/PabUNNuwYQOysrJYaqhXuN0aVFXpE/c+0RoaUPT0b9FYUtJuX8QrLyNw4kQJqVqJxkbUFhbi8oqVqDt+/Ppppc5+ndxUahQ/P0AIKD4+UCwWqHY7Bn/4ARRfW5cuDSci79LV92+POf3UE/X19aivr2957XQ6JaYhT2Ay9Z2181W7d9+y0ABA+cZNUktNxd//gbLsbDScOwfR0HB94+2K4E2nmVQ/vzbFMTj9CZgHDOiltETkLfrOb/BesHz5cgQFBbV8REVFyY5E1GV1p051uK/+9GnImmR17tiBq6tWoenKFYibTzF1JY+iAGZzm0Lj97OfoX8mr2oiojsndaZmwYIFWLFiRaefc+LECSQkJPTo+AsXLsQLL7zQ8trpdLLYkMfobObCNGCAtFNkZZs2X/9H81VM3eHjg4Gvr4RoaITb5YRtTCJso0be3YBE5LWklprf//73mDVrVqefExcX1+PjW61WWK2edTt6omb2xx5D2f+sbz29c5Og9HQJiQCtvh6NRUXXX5i7+etDVRGx6I+wP/ro3Q9GRATJpSY0NBShvMKB+pCmBjeOfHkBx78uQW1VI0wmBYEDfBGXOADRowbAMTRYtyzmkBBE/PkVXF7yKrTq6pbtARMnov/TT+mW42aKxQJTcBDcFZVQVBWK1Xr9uU03X5bdvBjY1xeKvz8UTQNUFb4jR6Df9OlSchORd/CYhcJFRUUoKytDUVER3G43CgoKAABDhw5FAG/IRXdBWUk1Pn+7AFVlrYvLmzSB8ks1yL9UhPycItgCfTDxt8MxeJQ+i1oDJkyA3z/+jqp9+6C5qmBLGgvrHcxe3ilFUWB/Ih3lGzcCAFR/f2iK0vJIA8VqhRoYCFFXB8VqvX6KTFWh2GwI/a/npeUmIu/gMZd0z5o1Cx988EG77Xv27MGDDz7YpWPwkm7qiBAC2944gktn2t8F+FYSxkdg0m9H9HKqvkk0NqJ0yauo2rOnZZsSGIDQ3/0OAQ89BFNAAKoPHkLl55/Bfe1HWOPjETz9V7AMGiQxNRF5MsPep+ZOsNRQR65ccOKz1QWor+7is54UIP35REQPD+ndYH1Y/blzqCsshGq3w3/CBKgWz3umFRF5Bt6nhqgb6qoau3eJtAAK91z06lJjjYuTeiqMiOinDH2fGqKuGhAVCLPZ1K2vaaht7KU0RETUEyw1RAD87BYkjI+Eau76vV+ivHiWhoioL2KpIboh5T/jcN8TsfCx3f6/haICI+936JCKiIi6imtqiG5QVQX3PjYY4x6JQeW1WpwruIrDX5xHY73W9hMVIOnhaNgCuTCWiKgvYakh+glFVRAc5oexadGoKK3B+cJrcDdqEJqAalYQPjgIyY/Hyo5JREQ/wVJD1AFVVfDgUwmILbyG7wt/hNakYdDw/hgyLhRmn+4tKiYiot7HUkPUCVVVEJsYithEPs6DiKiv40JhIiIiMgSWGiIiIjIElhoiIiIyBJYaIiIiMgSWGiIiIjIElhoiIiIyBJYaIiIiMgSWGiIiIjIElhoiIiIyBJYaIiIiMgSWGiIiIjIEr3r2kxACAOB0OiUnISIioq5qft9ufh/viFeVGpfLBQCIioqSnISIiIi6y+VyISgoqMP9irhd7TEQTdNQUlKCwMBAKIrS5a9zOp2IiopCcXEx7HZ7Lybse7x57ADH783j9+axAxy/N4+/L45dCAGXywWHwwFV7XjljFfN1KiqikGDBvX46+12e5/5AevNm8cOcPzePH5vHjvA8Xvz+Pva2DuboWnGhcJERERkCCw1REREZAgsNV1gtVqxePFiWK1W2VF0581jBzh+bx6/N48d4Pi9efyePHavWihMRERExsWZGiIiIjIElhoiIiIyBJYaIiIiMgSWGiIiIjIElpoOrFmzBmPGjGm5+VBqaip27twpO5Y0r732GhRFQVZWluwounj55ZehKEqbj4SEBNmxdHPx4kU89dRTCAkJgc1mw+jRo3H48GHZsXQxePDgdj97RVEwe/Zs2dF6ndvtxqJFixAbGwubzYYhQ4ZgyZIlt33ejpG4XC5kZWUhJiYGNpsN48ePR15enuxYveLrr79Geno6HA4HFEXBp59+2ma/EAJ/+tOfEBkZCZvNhrS0NJw+fVpO2C5iqenAoEGD8NprryE/Px+HDx/GxIkTMXXqVBw/flx2NN3l5eXhvffew5gxY2RH0dXIkSNx6dKllo99+/bJjqSL8vJyTJgwAT4+Pti5cye+++47vPHGG+jXr5/saLrIy8tr83PPzc0FAEyfPl1yst63YsUKrFmzBm+//TZOnDiBFStWYOXKlfjLX/4iO5punnvuOeTm5iI7OxuFhYV45JFHkJaWhosXL8qOdtdVV1cjMTER77zzzi33r1y5EqtXr8a7776LgwcPwt/fH48++ijq6up0TtoNgrqsX79+4m9/+5vsGLpyuVxi2LBhIjc3VzzwwANizpw5siPpYvHixSIxMVF2DCnmz58vfv7zn8uO0WfMmTNHDBkyRGiaJjtKr5syZYrIzMxss+2Xv/ylyMjIkJRIXzU1NcJkMont27e32T5u3Djx0ksvSUqlDwBi27ZtLa81TRMRERHi9ddfb9lWUVEhrFar+OijjyQk7BrO1HSB2+3Gli1bUF1djdTUVNlxdDV79mxMmTIFaWlpsqPo7vTp03A4HIiLi0NGRgaKiopkR9LFZ599huTkZEyfPh1hYWFISkrC+++/LzuWFA0NDdi4cSMyMzO79RBcTzV+/Hjs3r0bp06dAgAcPXoU+/btw+TJkyUn00dTUxPcbjd8fX3bbLfZbF4zU9vs/PnzKC0tbfO7PygoCCkpKThw4IDEZJ3zqgdadldhYSFSU1NRV1eHgIAAbNu2DSNGjJAdSzdbtmzBkSNHDHs+uTMpKSnYsGED4uPjcenSJbzyyiu4//77cezYMQQGBsqO16vOnTuHNWvW4IUXXsCLL76IvLw8PP/887BYLJg5c6bseLr69NNPUVFRgVmzZsmOoosFCxbA6XQiISEBJpMJbrcbS5cuRUZGhuxouggMDERqaiqWLFmC4cOHIzw8HB999BEOHDiAoUOHyo6nq9LSUgBAeHh4m+3h4eEt+/oilppOxMfHo6CgAJWVlfjkk08wc+ZM7N271yuKTXFxMebMmYPc3Nx2f7V4g5v/Mh0zZgxSUlIQExODrVu34tlnn5WYrPdpmobk5GQsW7YMAJCUlIRjx47h3Xff9bpSs27dOkyePBkOh0N2FF1s3boVmzZtwubNmzFy5EgUFBQgKysLDofDa3722dnZyMzMxMCBA2EymTBu3Dg8+eSTyM/Plx2NuoCnnzphsVgwdOhQ3HvvvVi+fDkSExPx1ltvyY6li/z8fFy5cgXjxo2D2WyG2WzG3r17sXr1apjNZrjdbtkRdRUcHIx77rkHZ86ckR2l10VGRrYr7sOHD/ea02/NLly4gF27duG5556THUU38+bNw4IFC/Cb3/wGo0ePxtNPP425c+di+fLlsqPpZsiQIdi7dy+qqqpQXFyMQ4cOobGxEXFxcbKj6SoiIgIAcPny5TbbL1++3LKvL2Kp6QZN01BfXy87hi4mTZqEwsJCFBQUtHwkJycjIyMDBQUFMJlMsiPqqqqqCmfPnkVkZKTsKL1uwoQJOHnyZJttp06dQkxMjKREcqxfvx5hYWGYMmWK7Ci6qampgaq2fVswmUzQNE1SInn8/f0RGRmJ8vJy5OTkYOrUqbIj6So2NhYRERHYvXt3yzan04mDBw/26bWlPP3UgYULF2Ly5MmIjo6Gy+XC5s2b8dVXXyEnJ0d2NF0EBgZi1KhRbbb5+/sjJCSk3XYj+sMf/oD09HTExMSgpKQEixcvhslkwpNPPik7Wq+bO3cuxo8fj2XLluHXv/41Dh06hLVr12Lt2rWyo+lG0zSsX78eM2fOhNnsPb8m09PTsXTpUkRHR2PkyJH49ttv8eabbyIzM1N2NN3k5ORACIH4+HicOXMG8+bNQ0JCAp555hnZ0e66qqqqNrPP58+fR0FBAfr374/o6GhkZWXh1VdfxbBhwxAbG4tFixbB4XBg2rRp8kLfjuzLr/qqzMxMERMTIywWiwgNDRWTJk0SX375pexYUnnTJd0zZswQkZGRwmKxiIEDB4oZM2aIM2fOyI6lm88//1yMGjVKWK1WkZCQINauXSs7kq5ycnIEAHHy5EnZUXTldDrFnDlzRHR0tPD19RVxcXHipZdeEvX19bKj6ebjjz8WcXFxwmKxiIiICDF79mxRUVEhO1av2LNnjwDQ7mPmzJlCiOuXdS9atEiEh4cLq9UqJk2a1Of/TyhCeNGtIomIiMiwuKaGiIiIDIGlhoiIiAyBpYaIiIgMgaWGiIiIDIGlhoiIiAyBpYaIiIgMgaWGiIiIDIGlhoiIiAyBpYaIiIgMgaWGiIiIDIGlhoiIiAyBpYaIPNbVq1cRERGBZcuWtWzbv38/LBYLdu/eLTEZEcnAB1oSkUfbsWMHpk2bhv379yM+Ph5jx47F1KlT8eabb8qORkQ6Y6khIo83e/Zs7Nq1C8nJySgsLEReXh6sVqvsWESkM5YaIvJ4tbW1GDVqFIqLi5Gfn4/Ro0fLjkREEnBNDRF5vLNnz6KkpASapuH777+XHYeIJOFMDRF5tIaGBtx3330YO3Ys4uPjsWrVKhQWFiIsLEx2NCLSGUsNEXm0efPm4ZNPPsHRo0cREBCABx54AEFBQdi+fbvsaESkM55+IiKP9dVXX2HVqlXIzs6G3W6HqqrIzs7GN998gzVr1siOR0Q640wNERERGQJnaoiIiMgQWGqIiIjIEFhqiIiIyBBYaoiIiMgQWGqIiIjIEFhqiIiIyBBYaoiIiMgQWGqIiIjIEFhqiIiIyBBYaoiIiMgQWGqIiIjIEP4fpiHdDD7Xkv0AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(data=to_plot, x='x', y='y', hue='cluster', linewidth=0, legend=False, s=30, alpha=0.9)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0f386147",
   "metadata": {},
   "source": [
    "### Inspecting the Clusters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "c7967fe8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--------------------\n",
      "Cluster 0\n",
      "Ver panel de control de los datos de entrada de RR. HH.\n",
      "eI-9 Dashboard\n",
      "Document Cloud Dashboard\n",
      "Registration Dashboard\n",
      "lmage Library\n",
      "Azure AD Dashboard\n",
      "Panel de control de registro\n",
      "Outbound Integrations Execution Dashboard\n",
      "Tableau de bord de la surveillance IDX Azure\n",
      "Access ADPRM Recruiting Dashboard\n",
      "--------------------\n",
      "Cluster 1\n",
      "Disciplinary Incidents\n",
      "View Scope Management\n",
      "View Your Benefits\n",
      "Ver mis identificaciones\n",
      "Gestion des attributions\n",
      "View Wisely\n",
      "Responsibilities\n",
      "Update Goals / Objectives\n",
      "Extended Benefits Enrolment\n",
      "Afficher et gérer les avantages sociaux du collaborateur\n",
      "--------------------\n",
      "Cluster 2\n",
      "Manage Associates' Pay Data\n",
      "View Your Pay\n",
      "Manage Payroll Policies\n",
      "Manage Compensation Reviews\n",
      "Configure Compensation Codes\n",
      "View My Pay Statements\n",
      "Pay Statement via Company Links\n",
      "Manage global payroll\n",
      "Ver su pago\n",
      "Configure Worker Compensation Codes\n",
      "--------------------\n",
      "Cluster 3\n",
      "Approve Time\n",
      "Afficher mon calendrier\n",
      "Enter Hours Worked\n",
      "Manage Timecards\n",
      "View Additional Time Features\n",
      "View/Edit or Approve Team Time Card\n",
      "Manage Time\n",
      "Clock In/Out\n",
      "Solicitar tiempo libre\n",
      "View My Timecard\n",
      "--------------------\n",
      "Cluster 4\n",
      "Afficher  l'organigramme de l'équipe\n",
      "View Team Check-Ins\n",
      "View Teams\n",
      "View Team Check-ins\n",
      "View Your Teams\n",
      "Afficher les horaires de l’équipe\n",
      "Manage Team Onboarding\n",
      "View My Snapshot\n",
      "Afficher et approuver les fiches de présence de l’équipe\n",
      "Afficher et approuver les congés de l’équipe\n",
      "--------------------\n",
      "Cluster 5\n",
      "Importer les données de configuration du système\n",
      "Importer les données du travailleur\n",
      "Core HR Inbound Import Staging List\n",
      "Export Associate Details\n",
      "Associate data import\n",
      "Associate Directory\n",
      "Importer les données du collaborateur\n",
      "Import Client Provided Translations\n",
      "Import Associate Data\n",
      "Payroll Imports\n",
      "--------------------\n",
      "Cluster 6\n",
      "View Job Chart\n",
      "Configurar los tipos de relaciones laborales\n",
      "View Job Templates\n",
      "Talent Management Demo Tools\n",
      "Configure Career Profile\n",
      "Afficher le tableau d'emplois\n",
      "Manage Advanced Workflows\n",
      "View Classic Recruiting Management\n",
      "Cessation d'emploi en masse\n",
      "Configure Unions\n",
      "--------------------\n",
      "Cluster -1\n",
      "TestLinkWithSequence3\n",
      "View Birthdays\n",
      "Afficher les anniversaires de travail\n",
      "Gérer les préférences de notification d'anniversaire/de consentement\n",
      "Create/Update/Delete Associate Data\n",
      "Complete Engagement Pulse\n",
      "Licenses & Certifications Dashboard\n",
      "Manage Broadcast(s) - Voice of the Employee\n",
      "Configure Approval Request Workflow\n",
      "Licences & Certifications Dashboard\n"
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
   "cell_type": "code",
   "execution_count": 37,
   "id": "3fa883b6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TestLinkWithSequence3\n",
      "View Birthdays\n",
      "Afficher les anniversaires de travail\n",
      "Gérer les préférences de notification d'anniversaire/de consentement\n",
      "Create/Update/Delete Associate Data\n",
      "Complete Engagement Pulse\n",
      "Licenses & Certifications Dashboard\n",
      "Manage Broadcast(s) - Voice of the Employee\n",
      "Configure Approval Request Workflow\n",
      "Licences & Certifications Dashboard\n",
      "actionlinkLFtest\n",
      "Outils de base pour l’intégration\n",
      "Work Authorization Type Configuration\n",
      "Manage Licenses and Certifications Library\n",
      "Gérer les autorisations\n",
      "View Team's Snapshot\n",
      "Work Authorisation Type Configuration\n",
      "View Criteria Control Center\n",
      "View Fluid Field Activity Logs\n",
      "Manage EV5 Integration\n"
     ]
    }
   ],
   "source": [
    "for index in np.where(labels==-1)[0][:20]:\n",
    "    print(caption[index])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3afb505",
   "metadata": {},
   "source": [
    "### Topic Modeling - Including Multilingual Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "b9e5393e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from bertopic import BERTopic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "959eb4bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "topic_model = BERTopic()\n",
    "topics, probs = topic_model.fit_transform(caption)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "9f3a0523",
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
       "      <th>Topic</th>\n",
       "      <th>Count</th>\n",
       "      <th>Name</th>\n",
       "      <th>Representation</th>\n",
       "      <th>Representative_Docs</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-1</td>\n",
       "      <td>38</td>\n",
       "      <td>-1_certifications_licenses_licences_de</td>\n",
       "      <td>[certifications, licenses, licences, de, modif...</td>\n",
       "      <td>[Manage My Licenses or Certifications, View My...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>278</td>\n",
       "      <td>0_view_manage_my_configure</td>\n",
       "      <td>[view, manage, my, configure, management, impo...</td>\n",
       "      <td>[View My Time Off, View and Manage Associates'...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>62</td>\n",
       "      <td>1_les_afficher_grer_de</td>\n",
       "      <td>[les, afficher, grer, de, et, sociaux, voir, a...</td>\n",
       "      <td>[Afficher et gérer les avantages sociaux, Affi...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>36</td>\n",
       "      <td>2_de_ver_mis_agregar</td>\n",
       "      <td>[de, ver, mis, agregar, datos, gestionar, los,...</td>\n",
       "      <td>[Panel de control de licencias y certificacion...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>35</td>\n",
       "      <td>3_dashboard_integration_integrations_document</td>\n",
       "      <td>[dashboard, integration, integrations, documen...</td>\n",
       "      <td>[Integration Dashboard, Manage eI-9 Dashboard,...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>4</td>\n",
       "      <td>20</td>\n",
       "      <td>4_tableau_bord_de_le</td>\n",
       "      <td>[tableau, bord, de, le, des, afficher, dataclo...</td>\n",
       "      <td>[Tableau de bord Datacloud de l'administrateur...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Topic  Count                                           Name  \\\n",
       "0     -1     38         -1_certifications_licenses_licences_de   \n",
       "1      0    278                     0_view_manage_my_configure   \n",
       "2      1     62                         1_les_afficher_grer_de   \n",
       "3      2     36                           2_de_ver_mis_agregar   \n",
       "4      3     35  3_dashboard_integration_integrations_document   \n",
       "5      4     20                           4_tableau_bord_de_le   \n",
       "\n",
       "                                      Representation  \\\n",
       "0  [certifications, licenses, licences, de, modif...   \n",
       "1  [view, manage, my, configure, management, impo...   \n",
       "2  [les, afficher, grer, de, et, sociaux, voir, a...   \n",
       "3  [de, ver, mis, agregar, datos, gestionar, los,...   \n",
       "4  [dashboard, integration, integrations, documen...   \n",
       "5  [tableau, bord, de, le, des, afficher, dataclo...   \n",
       "\n",
       "                                 Representative_Docs  \n",
       "0  [Manage My Licenses or Certifications, View My...  \n",
       "1  [View My Time Off, View and Manage Associates'...  \n",
       "2  [Afficher et gérer les avantages sociaux, Affi...  \n",
       "3  [Panel de control de licencias y certificacion...  \n",
       "4  [Integration Dashboard, Manage eI-9 Dashboard,...  \n",
       "5  [Tableau de bord Datacloud de l'administrateur...  "
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "topic_model.get_topic_info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "fece7396",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('certifications', 0.19536705930393858),\n",
       " ('licenses', 0.11605076345668046),\n",
       " ('licences', 0.11605076345668046),\n",
       " ('de', 0.11136139575885486),\n",
       " ('modifier', 0.11011194202444814),\n",
       " ('mes', 0.10120133257483101),\n",
       " ('performance', 0.07590099943112326),\n",
       " ('my', 0.072715338215232),\n",
       " ('du', 0.07094328869937537),\n",
       " ('delegates', 0.067299676846102)]"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "topic_model.get_topic(-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "5b1a2258",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('view', 0.1298164988678768),\n",
       " ('manage', 0.1045911326793215),\n",
       " ('my', 0.0787453906525927),\n",
       " ('configure', 0.06733083355317213),\n",
       " ('management', 0.048066049188960604),\n",
       " ('import', 0.04533393523156688),\n",
       " ('and', 0.04090477351522098),\n",
       " ('your', 0.040315697583632085),\n",
       " ('data', 0.03929516603610824),\n",
       " ('team', 0.034921086676854396)]"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "topic_model.get_topic(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "9f3310a7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('les', 0.23658689215180503),\n",
       " ('afficher', 0.1996201902530855),\n",
       " ('grer', 0.1371033978775642),\n",
       " ('de', 0.11863750024365267),\n",
       " ('et', 0.09606087965775054),\n",
       " ('sociaux', 0.07764549941086574),\n",
       " ('voir', 0.06767677499841497),\n",
       " ('avantages', 0.06767677499841497),\n",
       " ('des', 0.06596455789979146),\n",
       " ('congs', 0.06470458284238813)]"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "topic_model.get_topic(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "9cb5915e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('de', 0.24739291438768377),\n",
       " ('ver', 0.20013687049446433),\n",
       " ('mis', 0.12652614456225406),\n",
       " ('agregar', 0.08530698472546903),\n",
       " ('datos', 0.08530698472546903),\n",
       " ('gestionar', 0.08530698472546903),\n",
       " ('los', 0.08001015368752504),\n",
       " ('panel', 0.07591568673735244),\n",
       " ('control', 0.06977234730314436),\n",
       " ('la', 0.06521494861805935)]"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "topic_model.get_topic(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "4c9fa9d7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('dashboard', 0.6066743159255191),\n",
       " ('integration', 0.22472399348721722),\n",
       " ('integrations', 0.16514916338066066),\n",
       " ('document', 0.1566977636501762),\n",
       " ('ei9', 0.11752332273763215),\n",
       " ('cloud', 0.09577261705022207),\n",
       " ('hr', 0.08804118295384947),\n",
       " ('inbound', 0.08804118295384947),\n",
       " ('azure', 0.08804118295384947),\n",
       " ('datacloud', 0.08257458169033033)]"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "topic_model.get_topic(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "f6dd90c3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('tableau', 0.41509404727185845),\n",
       " ('bord', 0.3994822754815837),\n",
       " ('de', 0.30149060803007044),\n",
       " ('le', 0.17072531264239926),\n",
       " ('des', 0.16142546282794495),\n",
       " ('afficher', 0.09046322984971203),\n",
       " ('datacloud', 0.06981915850239313),\n",
       " ('gestion', 0.06624620902283872),\n",
       " ('ei9', 0.06624620902283872),\n",
       " ('du', 0.05690843754746643)]"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "topic_model.get_topic(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "f22d5642",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "details_search_value\n",
       "Test                     22305\n",
       "All Delegates            15382\n",
       "View People Movements    13238\n",
       "manage                   10432\n",
       "Teams                     9922\n",
       "                         ...  \n",
       "43132                        1\n",
       "242522                       1\n",
       "242032                       1\n",
       "242031                       1\n",
       "timw                         1\n",
       "Length: 4050, dtype: int64"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "filtered_actions = actions[~actions['locale'].isin(['fr-CA'])]\n",
    "filtered_actions.groupby(['details_search_value']).size().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "289b24ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "query = list(set(filtered_actions['details_search_value']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "35c044e7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "#T_8ca8f_row0_col0, #T_8ca8f_row0_col1, #T_8ca8f_row0_col2 {\n",
       "  background-color: #f7fbff;\n",
       "  color: #000000;\n",
       "  color: black;\n",
       "}\n",
       "</style>\n",
       "<table id=\"T_8ca8f\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_8ca8f_level0_col0\" class=\"col_heading level0 col0\" >my</th>\n",
       "      <th id=\"T_8ca8f_level0_col1\" class=\"col_heading level0 col1\" >action </th>\n",
       "      <th id=\"T_8ca8f_level0_col2\" class=\"col_heading level0 col2\" >link  </th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_8ca8f_level0_row0\" class=\"row_heading level0 row0\" >0_view_manage_my_configure</th>\n",
       "      <td id=\"T_8ca8f_row0_col0\" class=\"data row0 col0\" >0.111</td>\n",
       "      <td id=\"T_8ca8f_row0_col1\" class=\"data row0 col1\" >0.111</td>\n",
       "      <td id=\"T_8ca8f_row0_col2\" class=\"data row0 col2\" >0.111</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x125965c90>"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Calculate the topic distributions on a token-level\n",
    "topic_distr, topic_token_distr = topic_model.approximate_distribution('my action link', calculate_tokens=True)\n",
    "df = topic_model.visualize_approximate_distribution('my action link', topic_token_distr[0])\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "f9518b92",
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
         "customdata": [
          [
           0,
           "view | manage | my | configure | management",
           278
          ],
          [
           1,
           "les | afficher | grer | de | et",
           62
          ],
          [
           2,
           "de | ver | mis | agregar | datos",
           36
          ],
          [
           3,
           "dashboard | integration | integrations | document | ei9",
           35
          ],
          [
           4,
           "tableau | bord | de | le | des",
           20
          ]
         ],
         "hovertemplate": "<b>Topic %{customdata[0]}</b><br>%{customdata[1]}<br>Size: %{customdata[2]}",
         "legendgroup": "",
         "marker": {
          "color": "#B0BEC5",
          "line": {
           "color": "DarkSlateGrey",
           "width": 2
          },
          "size": [
           278,
           62,
           36,
           35,
           20
          ],
          "sizemode": "area",
          "sizeref": 0.17375,
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "",
         "orientation": "v",
         "showlegend": false,
         "type": "scatter",
         "x": [
          -6.290754795074463,
          -9.241832733154297,
          -9.380885124206543,
          -5.882449626922607,
          -9.940286636352539
         ],
         "xaxis": "x",
         "y": [
          24.881301879882812,
          8.46935749053955,
          7.783851146697998,
          24.47333526611328,
          8.648140907287598
         ],
         "yaxis": "y"
        }
       ],
       "layout": {
        "annotations": [
         {
          "showarrow": false,
          "text": "D1",
          "x": -11.43132963180542,
          "y": 17.614885318279267,
          "yshift": 10
         },
         {
          "showarrow": false,
          "text": "D2",
          "x": -8.215705907344818,
          "xshift": 10,
          "y": 28.613497161865233
         }
        ],
        "height": 650,
        "hoverlabel": {
         "bgcolor": "white",
         "font": {
          "family": "Rockwell",
          "size": 16
         }
        },
        "legend": {
         "itemsizing": "constant",
         "tracegroupgap": 0
        },
        "margin": {
         "t": 60
        },
        "shapes": [
         {
          "line": {
           "color": "#CFD8DC",
           "width": 2
          },
          "type": "line",
          "x0": -8.215705907344818,
          "x1": -8.215705907344818,
          "y0": 6.6162734746932985,
          "y1": 28.613497161865233
         },
         {
          "line": {
           "color": "#9E9E9E",
           "width": 2
          },
          "type": "line",
          "x0": -11.43132963180542,
          "x1": -5.000082182884216,
          "y0": 17.614885318279267,
          "y1": 17.614885318279267
         }
        ],
        "sliders": [
         {
          "active": 0,
          "pad": {
           "t": 50
          },
          "steps": [
           {
            "args": [
             {
              "marker.color": [
               [
                "red",
                "#B0BEC5",
                "#B0BEC5",
                "#B0BEC5",
                "#B0BEC5"
               ]
              ]
             }
            ],
            "label": "Topic 0",
            "method": "update"
           },
           {
            "args": [
             {
              "marker.color": [
               [
                "#B0BEC5",
                "red",
                "#B0BEC5",
                "#B0BEC5",
                "#B0BEC5"
               ]
              ]
             }
            ],
            "label": "Topic 1",
            "method": "update"
           },
           {
            "args": [
             {
              "marker.color": [
               [
                "#B0BEC5",
                "#B0BEC5",
                "red",
                "#B0BEC5",
                "#B0BEC5"
               ]
              ]
             }
            ],
            "label": "Topic 2",
            "method": "update"
           },
           {
            "args": [
             {
              "marker.color": [
               [
                "#B0BEC5",
                "#B0BEC5",
                "#B0BEC5",
                "red",
                "#B0BEC5"
               ]
              ]
             }
            ],
            "label": "Topic 3",
            "method": "update"
           },
           {
            "args": [
             {
              "marker.color": [
               [
                "#B0BEC5",
                "#B0BEC5",
                "#B0BEC5",
                "#B0BEC5",
                "red"
               ]
              ]
             }
            ],
            "label": "Topic 4",
            "method": "update"
           }
          ]
         }
        ],
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "rgb(36,36,36)"
            },
            "error_y": {
             "color": "rgb(36,36,36)"
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
             "endlinecolor": "rgb(36,36,36)",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "rgb(36,36,36)"
            },
            "baxis": {
             "endlinecolor": "rgb(36,36,36)",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "rgb(36,36,36)"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "colorscale": [
             [
              0,
              "#440154"
             ],
             [
              0.1111111111111111,
              "#482878"
             ],
             [
              0.2222222222222222,
              "#3e4989"
             ],
             [
              0.3333333333333333,
              "#31688e"
             ],
             [
              0.4444444444444444,
              "#26828e"
             ],
             [
              0.5555555555555556,
              "#1f9e89"
             ],
             [
              0.6666666666666666,
              "#35b779"
             ],
             [
              0.7777777777777778,
              "#6ece58"
             ],
             [
              0.8888888888888888,
              "#b5de2b"
             ],
             [
              1,
              "#fde725"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "colorscale": [
             [
              0,
              "#440154"
             ],
             [
              0.1111111111111111,
              "#482878"
             ],
             [
              0.2222222222222222,
              "#3e4989"
             ],
             [
              0.3333333333333333,
              "#31688e"
             ],
             [
              0.4444444444444444,
              "#26828e"
             ],
             [
              0.5555555555555556,
              "#1f9e89"
             ],
             [
              0.6666666666666666,
              "#35b779"
             ],
             [
              0.7777777777777778,
              "#6ece58"
             ],
             [
              0.8888888888888888,
              "#b5de2b"
             ],
             [
              1,
              "#fde725"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "colorscale": [
             [
              0,
              "#440154"
             ],
             [
              0.1111111111111111,
              "#482878"
             ],
             [
              0.2222222222222222,
              "#3e4989"
             ],
             [
              0.3333333333333333,
              "#31688e"
             ],
             [
              0.4444444444444444,
              "#26828e"
             ],
             [
              0.5555555555555556,
              "#1f9e89"
             ],
             [
              0.6666666666666666,
              "#35b779"
             ],
             [
              0.7777777777777778,
              "#6ece58"
             ],
             [
              0.8888888888888888,
              "#b5de2b"
             ],
             [
              1,
              "#fde725"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "line": {
              "color": "white",
              "width": 0.6
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "colorscale": [
             [
              0,
              "#440154"
             ],
             [
              0.1111111111111111,
              "#482878"
             ],
             [
              0.2222222222222222,
              "#3e4989"
             ],
             [
              0.3333333333333333,
              "#31688e"
             ],
             [
              0.4444444444444444,
              "#26828e"
             ],
             [
              0.5555555555555556,
              "#1f9e89"
             ],
             [
              0.6666666666666666,
              "#35b779"
             ],
             [
              0.7777777777777778,
              "#6ece58"
             ],
             [
              0.8888888888888888,
              "#b5de2b"
             ],
             [
              1,
              "#fde725"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "colorscale": [
             [
              0,
              "#440154"
             ],
             [
              0.1111111111111111,
              "#482878"
             ],
             [
              0.2222222222222222,
              "#3e4989"
             ],
             [
              0.3333333333333333,
              "#31688e"
             ],
             [
              0.4444444444444444,
              "#26828e"
             ],
             [
              0.5555555555555556,
              "#1f9e89"
             ],
             [
              0.6666666666666666,
              "#35b779"
             ],
             [
              0.7777777777777778,
              "#6ece58"
             ],
             [
              0.8888888888888888,
              "#b5de2b"
             ],
             [
              1,
              "#fde725"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
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
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "colorscale": [
             [
              0,
              "#440154"
             ],
             [
              0.1111111111111111,
              "#482878"
             ],
             [
              0.2222222222222222,
              "#3e4989"
             ],
             [
              0.3333333333333333,
              "#31688e"
             ],
             [
              0.4444444444444444,
              "#26828e"
             ],
             [
              0.5555555555555556,
              "#1f9e89"
             ],
             [
              0.6666666666666666,
              "#35b779"
             ],
             [
              0.7777777777777778,
              "#6ece58"
             ],
             [
              0.8888888888888888,
              "#b5de2b"
             ],
             [
              1,
              "#fde725"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "rgb(237,237,237)"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "rgb(217,217,217)"
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
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 1,
            "tickcolor": "rgb(36,36,36)",
            "ticks": "outside"
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "rgb(103,0,31)"
            ],
            [
             0.1,
             "rgb(178,24,43)"
            ],
            [
             0.2,
             "rgb(214,96,77)"
            ],
            [
             0.3,
             "rgb(244,165,130)"
            ],
            [
             0.4,
             "rgb(253,219,199)"
            ],
            [
             0.5,
             "rgb(247,247,247)"
            ],
            [
             0.6,
             "rgb(209,229,240)"
            ],
            [
             0.7,
             "rgb(146,197,222)"
            ],
            [
             0.8,
             "rgb(67,147,195)"
            ],
            [
             0.9,
             "rgb(33,102,172)"
            ],
            [
             1,
             "rgb(5,48,97)"
            ]
           ],
           "sequential": [
            [
             0,
             "#440154"
            ],
            [
             0.1111111111111111,
             "#482878"
            ],
            [
             0.2222222222222222,
             "#3e4989"
            ],
            [
             0.3333333333333333,
             "#31688e"
            ],
            [
             0.4444444444444444,
             "#26828e"
            ],
            [
             0.5555555555555556,
             "#1f9e89"
            ],
            [
             0.6666666666666666,
             "#35b779"
            ],
            [
             0.7777777777777778,
             "#6ece58"
            ],
            [
             0.8888888888888888,
             "#b5de2b"
            ],
            [
             1,
             "#fde725"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#440154"
            ],
            [
             0.1111111111111111,
             "#482878"
            ],
            [
             0.2222222222222222,
             "#3e4989"
            ],
            [
             0.3333333333333333,
             "#31688e"
            ],
            [
             0.4444444444444444,
             "#26828e"
            ],
            [
             0.5555555555555556,
             "#1f9e89"
            ],
            [
             0.6666666666666666,
             "#35b779"
            ],
            [
             0.7777777777777778,
             "#6ece58"
            ],
            [
             0.8888888888888888,
             "#b5de2b"
            ],
            [
             1,
             "#fde725"
            ]
           ]
          },
          "colorway": [
           "#1F77B4",
           "#FF7F0E",
           "#2CA02C",
           "#D62728",
           "#9467BD",
           "#8C564B",
           "#E377C2",
           "#7F7F7F",
           "#BCBD22",
           "#17BECF"
          ],
          "font": {
           "color": "rgb(36,36,36)"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "white",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
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
            "gridcolor": "rgb(232,232,232)",
            "linecolor": "rgb(36,36,36)",
            "showgrid": false,
            "showline": true,
            "ticks": "outside"
           },
           "bgcolor": "white",
           "radialaxis": {
            "gridcolor": "rgb(232,232,232)",
            "linecolor": "rgb(36,36,36)",
            "showgrid": false,
            "showline": true,
            "ticks": "outside"
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "white",
            "gridcolor": "rgb(232,232,232)",
            "gridwidth": 2,
            "linecolor": "rgb(36,36,36)",
            "showbackground": true,
            "showgrid": false,
            "showline": true,
            "ticks": "outside",
            "zeroline": false,
            "zerolinecolor": "rgb(36,36,36)"
           },
           "yaxis": {
            "backgroundcolor": "white",
            "gridcolor": "rgb(232,232,232)",
            "gridwidth": 2,
            "linecolor": "rgb(36,36,36)",
            "showbackground": true,
            "showgrid": false,
            "showline": true,
            "ticks": "outside",
            "zeroline": false,
            "zerolinecolor": "rgb(36,36,36)"
           },
           "zaxis": {
            "backgroundcolor": "white",
            "gridcolor": "rgb(232,232,232)",
            "gridwidth": 2,
            "linecolor": "rgb(36,36,36)",
            "showbackground": true,
            "showgrid": false,
            "showline": true,
            "ticks": "outside",
            "zeroline": false,
            "zerolinecolor": "rgb(36,36,36)"
           }
          },
          "shapedefaults": {
           "fillcolor": "black",
           "line": {
            "width": 0
           },
           "opacity": 0.3
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "rgb(232,232,232)",
            "linecolor": "rgb(36,36,36)",
            "showgrid": false,
            "showline": true,
            "ticks": "outside"
           },
           "baxis": {
            "gridcolor": "rgb(232,232,232)",
            "linecolor": "rgb(36,36,36)",
            "showgrid": false,
            "showline": true,
            "ticks": "outside"
           },
           "bgcolor": "white",
           "caxis": {
            "gridcolor": "rgb(232,232,232)",
            "linecolor": "rgb(36,36,36)",
            "showgrid": false,
            "showline": true,
            "ticks": "outside"
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "rgb(232,232,232)",
           "linecolor": "rgb(36,36,36)",
           "showgrid": false,
           "showline": true,
           "ticks": "outside",
           "title": {
            "standoff": 15
           },
           "zeroline": false,
           "zerolinecolor": "rgb(36,36,36)"
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "rgb(232,232,232)",
           "linecolor": "rgb(36,36,36)",
           "showgrid": false,
           "showline": true,
           "ticks": "outside",
           "title": {
            "standoff": 15
           },
           "zeroline": false,
           "zerolinecolor": "rgb(36,36,36)"
          }
         }
        },
        "title": {
         "font": {
          "color": "Black",
          "size": 22
         },
         "text": "<b>Intertopic Distance Map</b>",
         "x": 0.5,
         "xanchor": "center",
         "y": 0.95,
         "yanchor": "top"
        },
        "width": 650,
        "xaxis": {
         "anchor": "y",
         "domain": [
          0,
          1
         ],
         "range": [
          -11.43132963180542,
          -5.000082182884216
         ],
         "title": {
          "text": ""
         },
         "visible": false
        },
        "yaxis": {
         "anchor": "x",
         "domain": [
          0,
          1
         ],
         "range": [
          6.6162734746932985,
          28.613497161865233
         ],
         "title": {
          "text": ""
         },
         "visible": false
        }
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "topic_model.visualize_topics()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "9c6f6bf3",
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
         "hoverinfo": "text",
         "hovertext": [
          "Configurer les buts/objectifs",
          "Licenses & Certifications Dashboard",
          "Modifier mes données démographiques personnelles",
          "Configuration du type de permis de travail",
          "Lifion Generative AI",
          "Manage Licences and Certifications Library",
          "Licences & Certifications Dashboard",
          "Importer les données de configuration du système",
          "Responsabilités",
          "Gestion des attributions",
          "Manage My Licenses or Certifications",
          "Responsabilidades",
          "Configurar los tipos de relaciones laborales",
          "Ajouter mon adresse physique",
          "Gestion des actifs",
          "Configurar reconocimientos",
          "Performance Setup",
          "All Delegates",
          "Afficher l'évaluation de votre rendement",
          "AOID Tracker",
          "My Delegates Information",
          "View My Licenses & Certifications",
          "View My Licences & Certifications",
          "Configurer les congés autorisés",
          "Modifier l'accès aux produits",
          "Modifier mes nationalités",
          "Mettre à jour mon adresse de courriel",
          "Mettre à jour mes numéros de téléphone",
          "Afficher le centre de contrôle des critères",
          "Performance Evaluation",
          "Manage My Licences or Certifications",
          "DHModalsize",
          "Modifier l’image du profil",
          "Configure Performance Plans",
          "Responsibilities",
          "Ver el panel de control de sincronización de empleado de Cornerstone",
          "Manage Licenses and Certifications Library",
          "Afficher mes enregistrements",
          null
         ],
         "marker": {
          "color": "#CFD8DC",
          "opacity": 0.5,
          "size": 5
         },
         "mode": "markers+text",
         "name": "other",
         "showlegend": false,
         "type": "scattergl",
         "x": [
          5.433406352996826,
          -0.6969897747039795,
          5.143036842346191,
          4.918051242828369,
          4.909918785095215,
          -0.7470312118530273,
          -0.7068042159080505,
          5.309678554534912,
          4.126065254211426,
          5.263594150543213,
          -0.7335953712463379,
          4.154952526092529,
          4.011157512664795,
          5.418914794921875,
          5.271745681762695,
          3.9858415126800537,
          3.007866144180298,
          4.86965274810791,
          5.211612701416016,
          2.390662431716919,
          4.885817527770996,
          -0.6983606219291687,
          -0.7000911235809326,
          5.752264499664307,
          5.171907424926758,
          5.140471458435059,
          5.561395645141602,
          5.580151557922363,
          5.109748840332031,
          3.0273401737213135,
          -0.7322723865509033,
          5.154909133911133,
          5.173945426940918,
          3.046403646469116,
          4.1159796714782715,
          3.0857646465301514,
          -0.726830780506134,
          4.330075740814209,
          3.495272159576416
         ],
         "y": [
          13.48580551147461,
          11.746294021606445,
          11.165600776672363,
          13.611438751220703,
          10.619643211364746,
          11.693596839904785,
          11.740464210510254,
          13.827455520629883,
          11.368021965026855,
          13.480501174926758,
          11.688154220581055,
          11.415514945983887,
          11.782265663146973,
          12.69366455078125,
          13.485124588012695,
          11.571420669555664,
          5.6923675537109375,
          10.934149742126465,
          11.890923500061035,
          12.496535301208496,
          10.894800186157227,
          11.746864318847656,
          11.744451522827148,
          13.833000183105469,
          11.210722923278809,
          11.12746524810791,
          13.039138793945312,
          13.060640335083008,
          13.940381050109863,
          5.7152886390686035,
          11.706522941589355,
          11.206192016601562,
          11.123064994812012,
          5.7341156005859375,
          11.18234920501709,
          11.686728477478027,
          11.713419914245605,
          13.59559440612793,
          11.596044540405273
         ]
        },
        {
         "hoverinfo": "text",
         "hovertext": [
          "Core HR Inbound Import Staging List",
          "View and Request Leave of Absence",
          "View Wisely-Flow",
          "View and Manage Associates' Benefits",
          "View Team Check-ins",
          "Broadcast Message(s) - Voice of the Employee",
          "Manage Settings",
          "View Team Schedules",
          "View Everyone's OKRs",
          "View And Manage Fluid Fields",
          "Manage Users",
          "Add My Mailing Address",
          "Configure Associate Classification",
          "View Pulses To Complete",
          "Performance Settings",
          "Add New Hire",
          "Manage/Update Talent Assessments",
          "Payroll Imports",
          "View Engagement Pulse Results",
          "Manage Payroll Policies",
          "Configure Unions",
          "View Occupational Classifications",
          "Manage EEO Reports",
          "View/Edit or Approve Team Time Card",
          "Image Library",
          "View Cost Centres",
          "Manage Consent for ADP Benchmarks",
          "Organization Management",
          "View Work Anniversaries",
          "View Your Company's News",
          "View Enterprise Units",
          "View Criteria Control Center",
          "Bulk Import Custom Fields",
          "View My SnapShot",
          "View Company News",
          "Manage Timecards",
          "Manage Payload Configuration",
          "Configure Worker Compensation Codes",
          "Associate Directory",
          "Import Associate Data",
          "View Recruiting Overview",
          "View StandOut Suite",
          "Access Internal Career/Job Site",
          "Manage Return to Workplace Surveys",
          "Complete My Return To Workplace Survey",
          "Manage Disciplinary Actions",
          "Import Worker Data",
          "Manage Your Company's Compensation",
          "Configure Career Development",
          "Review My Skills",
          "Records of Employment",
          "Import System Configuration Data",
          "Termination - Configuration Settings",
          "Identify Critical Positions",
          "Manage Journeys",
          "View Your Performance Reviews",
          "View Your Dynamic Teams",
          "Manage Succession Plans",
          "View My Tax Withholding",
          "Manage Illness and Injury Reporting",
          "Manage My OKRs",
          "View Client Administrators",
          "View Company Org Chart",
          "View My Direct Deposits",
          "View Teams",
          "Process Internal Hires",
          "View and Approve Team Time Off",
          "User Management Troubleshooting",
          "Submit a help ticket",
          "View People Movements",
          "Recruiting Configuration",
          "Organisation Management",
          "Configure Talent Cycles",
          "View Job Templates",
          "View Birthdays",
          "Manage Successors for Positions",
          "Create Requisition Request",
          "Case Management",
          "Manage Compensation Reviews",
          "Edit Profile Image",
          "Configure Approval Request Workflow",
          "View organization information for Salvador Dali",
          "My Team Learning",
          "View Wisely",
          "Benefits Data Management",
          "Associate data import",
          "View My Associate Profile",
          "Pay Statement via Company Links",
          "Manage Team Onboarding",
          "Manage Birthday Notification/Consent Preferences",
          "View Positions",
          "Configure Work Authorization Expiration Reminders",
          "Manage Associates' Pay Data",
          "disabled TALACQ-55299",
          "Request Time Off",
          "Manage Recruiting",
          "Manage Talent Cycles",
          "lmage Library",
          "Configure Work Relationship Types",
          "Manage Benefits forms and plan documents",
          "View Your Surveys",
          "Extended Benefits Enrollment",
          "Ver registros de actividad de valores de campo fluido",
          "ESS Training",
          "Bulk Leave of Absence",
          "Manage global payroll",
          "Create/Update/Delete Associate Data",
          "View Leave Requests & Balances",
          "View Team Check-Ins",
          "Configure Recruiting",
          "Survey(s) - Voice of the Employee",
          "Configure Journeys",
          "Send Registration Emails",
          "Edit My Personal Demographics",
          "Recognize Someone",
          "View and Approve Team Timecards",
          "Configure Competencies",
          "Google",
          "Award Management",
          "Bulk Unit Management",
          "Configure Additional Associate Identifiers",
          "Update My Phone Numbers",
          "Performance Configuration",
          "Quarter and Year End Checklist",
          "View Fluid Field Activity Logs",
          "Configure Compensation Reviews",
          "Extended Benefits Enrolment",
          "View Time Reports",
          "Yahoo",
          "Manage Survey(s) - Voice of the Employee",
          "Clock In/Out",
          "Assign Additional Associate Identifiers",
          "Add My Physical Address",
          "Recognise Someone",
          "vamsi demo link",
          "Generate Cornerstone Files Manually",
          "Bulk Job Change",
          "View Client Provided Translations",
          "Import New Hires from ATS",
          "View FAQs",
          "View My Snapshot",
          "Disciplinary Incidents",
          "Configure Follow-up Actions",
          "Import Education History",
          "Asset Management",
          "Enter Hours Worked",
          "Create Position",
          "TestLinkWithSequence3",
          "View organization information for Pro Check One",
          "Manage Pay Group Transfer",
          "Manage your Successors",
          "Launch Engagement Pulse",
          "Manage Pay Runs",
          "Configure Onboarding",
          "Manage Payroll Company Data",
          "Create a Survey",
          "Change Login Methods",
          "Manage Journeys Deprecated",
          "View Your Onboarding",
          "View Additional Time Features",
          "View My Schedule",
          "Manage Time",
          "Reports & Analytics",
          "Benefits Implementation",
          "travel blog",
          "Manage Record of Employment",
          "View My Check-In",
          "View Completed Engagement Pulses",
          "My Learning",
          "Talent Management Demo Tools",
          "View Classic Recruiting Management",
          "Configure Unions and Bargaining Units",
          "Manage Fluid Field Definitions",
          "Export Current Data",
          "View My Org",
          "View Cost Centers",
          "Manage Cornerstone Configuration",
          "Manage Leave",
          "View Dynamic Teams Chart",
          "Import Legacy System Data",
          "Create Criteria",
          "Manage Cycle Setup",
          "View My Timecard",
          "Configure Career Profile",
          "One LMS Test",
          "Work Authorization Type Configuration",
          "View Criteria Control Centre",
          "Import New Hires",
          "Edit My Nationalities",
          "Complete Check-In",
          "Add My Personal Contacts",
          "Audit Reporting",
          "Update My Email",
          "Manage Broadcast(s) - Voice of the Employee",
          "Configure Advanced Workflows",
          "View My Check-ins",
          "Reporting Help",
          "Manage Company Time Off",
          "Import Client Provided Translations",
          "Manage New Hires from ATS",
          "Calibrate Talent Attributes",
          "View Job Chart",
          "My Courses",
          "Create Cycle",
          "Retrospective",
          "Manage Authorisations",
          "View support tickets",
          "View Dynamic Teams",
          "Go to BVCore",
          "Take my Strengths Assessment",
          "Create News Post",
          "Configure Goals / Objectives",
          "View My Time Off",
          "View Team Chart",
          "Manage My Career Development",
          "Search For Jobs",
          "Learning Management",
          "StandOut Champion Central",
          "Import Pay Data",
          "View Internal Career Site",
          "View Your Pay",
          "actionlinkLFtest",
          "View My Identifications",
          "Manage Advanced Workflows",
          "Compliance on Demand",
          "Manage VETS 4212 Reports",
          "Submit a Performance Pulse",
          "Configure Compensation Codes",
          "Create New Client",
          "Manage Hours and Earnings",
          "View Office Locations",
          "Configure Leave of Absence",
          "Configure Work Authorisation Expiration Reminders",
          "Complete Check-in",
          "Edit My Identification",
          "Company Links and FAQs",
          "Manage General Ledger",
          "View Engagement Pulse",
          "Change Product Access",
          "View Team's Snapshot",
          "Configurer la classification des associés",
          "Configure Recognitions",
          "View Scope Management",
          "Policy Management",
          "Prior Quarter Adjustment",
          "View My Pay Statements",
          "Pick up Shift",
          "Export Associate Details",
          "View and Manage Benefits",
          "Complete Engagement Pulse",
          "View My Reports' OKRs",
          "View Your Teams",
          "Access Absorb LMS",
          "Import Fluid Field Values",
          "View My Recognitions",
          "Manage Company Policies",
          "Work Authorisation Type Configuration",
          "Manage Central Rate",
          "Bulk Termination",
          "Manage Internal Mobility Pulse",
          "View Job Families",
          "Manage Authorizations",
          "Classroom Training Management",
          "Manage Payroll Calendar",
          "View My Check-Ins",
          "Manage News Posts",
          "View Team's SnapShot",
          "View Your Benefits",
          "View Client Management",
          "Approve Time",
          "Update Goals / Objectives",
          "Manage Benefits Supplemental Fields",
          "Configure Movement Reasons",
          "Edit Birth Details",
          "View Legal Entities",
          "Add My Postal Address",
          "Sync Coded Fields Table",
          "View Your Compensation",
          null
         ],
         "marker": {
          "opacity": 0.5,
          "size": 5
         },
         "mode": "markers+text",
         "name": "0_view_manage_my",
         "text": [
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "0_view_manage_my"
         ],
         "textfont": {
          "size": 12
         },
         "type": "scattergl",
         "x": [
          4.49263858795166,
          -0.16216962039470673,
          2.1554367542266846,
          2.7945544719696045,
          0.8328420519828796,
          0.09446421265602112,
          2.76082181930542,
          0.26833420991897583,
          1.0419151782989502,
          4.795047283172607,
          2.615856170654297,
          6.216968059539795,
          5.015652179718018,
          2.468381404876709,
          2.9700229167938232,
          4.169301986694336,
          3.350369453430176,
          4.370793342590332,
          2.479083776473999,
          1.9400068521499634,
          2.134702205657959,
          4.809407711029053,
          1.0697025060653687,
          -0.2306063324213028,
          4.779969215393066,
          2.2660889625549316,
          2.724973678588867,
          2.747093677520752,
          0.29552018642425537,
          1.2628936767578125,
          0.9819527864456177,
          2.3021068572998047,
          4.694079875946045,
          0.39090901613235474,
          1.2493772506713867,
          -0.26197969913482666,
          2.9273369312286377,
          1.76532781124115,
          5.0366411209106445,
          4.912155628204346,
          3.604926824569702,
          2.108215570449829,
          3.8688573837280273,
          0.025263851508498192,
          0.028858505189418793,
          1.6465771198272705,
          4.6018385887146,
          1.6040674448013306,
          3.7010691165924072,
          3.732758045196533,
          1.6980040073394775,
          4.647955894470215,
          2.543335199356079,
          1.9658358097076416,
          1.55597722530365,
          2.988633394241333,
          0.5728508234024048,
          2.762593984603882,
          1.2195862531661987,
          1.279145359992981,
          1.0411512851715088,
          3.3543646335601807,
          1.0302948951721191,
          1.1936535835266113,
          0.6042935252189636,
          4.1749091148376465,
          -0.2607572078704834,
          2.732708215713501,
          1.8133447170257568,
          2.1612305641174316,
          3.6393630504608154,
          2.7473204135894775,
          3.197357654571533,
          1.427116870880127,
          0.3474296033382416,
          2.779346466064453,
          0.19191870093345642,
          2.688110828399658,
          1.5794785022735596,
          5.133810043334961,
          2.087162494659424,
          1.0203967094421387,
          3.7840676307678223,
          2.0571770668029785,
          2.8333959579467773,
          4.757778644561768,
          5.0341644287109375,
          1.5982669591903687,
          3.298382043838501,
          2.610299825668335,
          2.0095648765563965,
          2.096888542175293,
          1.7653380632400513,
          -0.2663932740688324,
          -0.25882014632225037,
          3.572195291519165,
          3.2021474838256836,
          4.743083953857422,
          2.177652597427368,
          2.870194911956787,
          -0.003332689171656966,
          2.861654758453369,
          4.8066511154174805,
          3.5169856548309326,
          -0.15280425548553467,
          1.9529517889022827,
          5.004999160766602,
          -0.12664572894573212,
          0.8550559282302856,
          3.6051735877990723,
          0.06488993018865585,
          1.558689832687378,
          6.183453559875488,
          5.411818504333496,
          5.112894058227539,
          -0.2222478687763214,
          3.5102016925811768,
          4.12420654296875,
          2.8554434776306152,
          2.835144281387329,
          5.074197769165039,
          6.184887886047363,
          2.988027334213257,
          0.5999659895896912,
          4.768222808837891,
          1.536535382270813,
          2.867957830429077,
          -0.11677782237529755,
          4.144996166229248,
          0.04246705770492554,
          -0.27439606189727783,
          5.088396072387695,
          6.267918586730957,
          5.1984076499938965,
          3.110938549041748,
          2.835212230682373,
          4.1365742683410645,
          3.4452669620513916,
          4.384175777435303,
          1.5730955600738525,
          0.3780422806739807,
          1.4487882852554321,
          2.8808395862579346,
          4.612581253051758,
          2.713810443878174,
          -0.14376327395439148,
          2.0548598766326904,
          1.657365322113037,
          0.9627496600151062,
          1.7627815008163452,
          2.777031421661377,
          2.487501621246338,
          1.7946709394454956,
          3.316817045211792,
          1.816870927810669,
          0.029317060485482216,
          2.3038666248321533,
          1.5145460367202759,
          3.2694520950317383,
          -0.11596278101205826,
          0.071278877556324,
          -0.1911037564277649,
          1.0497875213623047,
          2.8287458419799805,
          1.53412663936615,
          1.6921558380126953,
          0.8891341090202332,
          2.469839096069336,
          3.844801187515259,
          3.229736328125,
          3.634488821029663,
          2.0810599327087402,
          4.80764627456665,
          4.871614933013916,
          1.0009682178497314,
          2.243861675262451,
          2.8175437450408936,
          -0.13899913430213928,
          0.7008848190307617,
          4.7778801918029785,
          2.251255989074707,
          2.8671224117279053,
          -0.23316825926303864,
          3.8489749431610107,
          2.494373321533203,
          2.1737120151519775,
          2.301168203353882,
          4.404857158660889,
          5.40578031539917,
          0.8243435621261597,
          6.248348712921143,
          1.1103535890579224,
          6.202490329742432,
          0.09590265899896622,
          2.15244197845459,
          0.903712272644043,
          1.1094509363174438,
          -0.2332811951637268,
          4.402095794677734,
          4.085942268371582,
          3.337350368499756,
          1.2150019407272339,
          3.8100485801696777,
          2.8360586166381836,
          0.1430087387561798,
          2.325871229171753,
          2.019792318344116,
          0.5759219527244568,
          4.561652183532715,
          3.6304104328155518,
          1.3027944564819336,
          3.1600069999694824,
          -0.2499702274799347,
          0.7188178300857544,
          3.658437728881836,
          4.091925621032715,
          3.3544301986694336,
          0.5795713663101196,
          4.623973846435547,
          3.807953119277954,
          1.3455287218093872,
          1.778597354888916,
          5.140048503875732,
          2.1375010013580322,
          2.6125545501708984,
          1.1001018285751343,
          2.7883920669555664,
          1.7501187324523926,
          3.5843868255615234,
          -0.09877219796180725,
          2.0094611644744873,
          -0.17608144879341125,
          2.0351498126983643,
          0.8213211297988892,
          5.333532333374023,
          1.4901316165924072,
          1.9638652801513672,
          2.496431827545166,
          2.2991018295288086,
          0.39824554324150085,
          5.054609298706055,
          4.92232608795166,
          2.6398000717163086,
          2.681572198867798,
          0.5398751497268677,
          1.2854596376419067,
          -0.059565309435129166,
          5.074811935424805,
          2.705388069152832,
          2.4806439876556396,
          1.0482475757598877,
          0.6147516965866089,
          2.4058732986450195,
          4.806288242340088,
          4.984508037567139,
          2.619572162628174,
          2.1779000759124756,
          1.9577269554138184,
          2.6451592445373535,
          2.4402356147766113,
          1.4599041938781738,
          2.337981939315796,
          3.4919521808624268,
          1.9518260955810547,
          0.902701199054718,
          1.2993038892745972,
          0.3940731883049011,
          2.8037989139556885,
          3.260554075241089,
          -0.264731228351593,
          3.2190089225769043,
          2.867570638656616,
          3.0050790309906006,
          5.364608287811279,
          0.9936479926109314,
          6.227105140686035,
          3.5892488956451416,
          1.3992880582809448,
          2.441220283508301
         ],
         "y": [
          8.328591346740723,
          8.71282958984375,
          7.012362480163574,
          7.46886682510376,
          7.957448482513428,
          9.591017723083496,
          10.12869644165039,
          7.38134765625,
          9.68838119506836,
          7.494961738586426,
          9.971233367919922,
          10.338557243347168,
          9.020078659057617,
          5.253840446472168,
          5.650539398193359,
          8.399036407470703,
          9.323363304138184,
          8.274677276611328,
          5.270000457763672,
          9.035387992858887,
          9.998360633850098,
          9.109972953796387,
          9.740703582763672,
          7.98724889755249,
          10.196893692016602,
          7.35997200012207,
          10.451011657714844,
          9.598645210266113,
          7.6756672859191895,
          7.544193744659424,
          7.337775230407715,
          7.356123447418213,
          8.073108673095703,
          6.722090244293213,
          7.454981327056885,
          8.005964279174805,
          11.051630020141602,
          9.389747619628906,
          8.666213989257812,
          8.476753234863281,
          8.801057815551758,
          7.067532539367676,
          8.99492359161377,
          9.587871551513672,
          9.561429023742676,
          9.504070281982422,
          8.252732276916504,
          9.04847526550293,
          9.129725456237793,
          9.670453071594238,
          8.986639022827148,
          8.335657119750977,
          10.351431846618652,
          6.864418029785156,
          10.76147747039795,
          5.683084011077881,
          7.096132755279541,
          8.98348617553711,
          8.425894737243652,
          9.59325885772705,
          9.715873718261719,
          7.82512092590332,
          7.312564849853516,
          8.41905403137207,
          7.120911121368408,
          8.50418758392334,
          8.184036254882812,
          9.829312324523926,
          7.936722278594971,
          6.623107433319092,
          8.810894966125488,
          9.656715393066406,
          9.180694580078125,
          7.2632880210876465,
          7.663869380950928,
          8.960238456726074,
          9.115376472473145,
          9.709986686706543,
          9.260221481323242,
          9.797608375549316,
          10.320420265197754,
          7.5035881996154785,
          9.785351753234863,
          7.077870845794678,
          7.479158401489258,
          8.314067840576172,
          8.680340766906738,
          8.498071670532227,
          8.563998222351074,
          10.529903411865234,
          6.867072582244873,
          10.36413288116455,
          8.715605735778809,
          8.519112586975098,
          8.416101455688477,
          8.816917419433594,
          9.182806968688965,
          10.323952674865723,
          10.201740264892578,
          7.444305896759033,
          9.527081489562988,
          7.473018646240234,
          7.505304336547852,
          9.648141860961914,
          8.755252838134766,
          8.927998542785645,
          8.62032413482666,
          8.733405113220215,
          7.929856777191162,
          8.826486587524414,
          9.588668823242188,
          10.789189338684082,
          10.378222465515137,
          9.750883102416992,
          9.460536003112793,
          8.004947662353516,
          9.238486289978027,
          9.945218086242676,
          9.529497146606445,
          9.56842041015625,
          8.734498023986816,
          10.348339080810547,
          5.671882152557373,
          8.139791488647461,
          7.471290111541748,
          9.297986030578613,
          7.476778507232666,
          7.894941806793213,
          9.986671447753906,
          9.574453353881836,
          8.149055480957031,
          8.753479957580566,
          10.260340690612793,
          9.430768013000488,
          9.173004150390625,
          11.344491958618164,
          8.50599479675293,
          7.739850044250488,
          8.284526824951172,
          7.654917240142822,
          6.7022223472595215,
          9.594978332519531,
          10.566322326660156,
          8.244553565979004,
          9.649026870727539,
          8.328428268432617,
          6.983111381530762,
          8.365589141845703,
          7.625561237335205,
          8.69688892364502,
          9.074142456054688,
          5.272953033447266,
          8.722168922424316,
          8.584391593933105,
          8.982165336608887,
          9.553040504455566,
          10.220084190368652,
          10.814839363098145,
          8.549165725708008,
          7.896347522735596,
          7.686707973480225,
          8.26041316986084,
          9.763065338134766,
          7.502073287963867,
          10.791101455688477,
          9.012174606323242,
          8.098777770996094,
          5.261551856994629,
          9.827898979187012,
          9.240270614624023,
          8.82854175567627,
          9.870567321777344,
          7.507205486297607,
          8.308016777038574,
          7.368594646453857,
          7.369336128234863,
          11.236044883728027,
          8.776611328125,
          7.127469062805176,
          8.302244186401367,
          7.358743190765381,
          8.864913940429688,
          7.924341678619385,
          9.05432415008545,
          9.303814888000488,
          10.298810958862305,
          7.340396404266357,
          8.326059341430664,
          9.71690845489502,
          8.068965911865234,
          10.304217338562012,
          9.702260971069336,
          10.331534385681152,
          9.558938980102539,
          10.315231323242188,
          8.028719902038574,
          9.69594955444336,
          8.360980033874512,
          8.02911376953125,
          8.42577838897705,
          9.307417869567871,
          7.183948993682861,
          9.812112808227539,
          8.880627632141113,
          7.9679789543151855,
          10.254678726196289,
          7.464164733886719,
          7.112866401672363,
          10.268771171569824,
          9.588523864746094,
          7.543532371520996,
          10.649042129516602,
          8.173571586608887,
          7.1053242683410645,
          9.21082878112793,
          9.877266883850098,
          9.572165489196777,
          7.066969394683838,
          8.217550277709961,
          9.029189109802246,
          8.518646240234375,
          8.433237075805664,
          9.569305419921875,
          10.2633638381958,
          9.656899452209473,
          9.69886302947998,
          5.484555721282959,
          9.375624656677246,
          7.849354267120361,
          8.383984565734863,
          7.1680450439453125,
          8.75317668914795,
          10.379721641540527,
          8.09487247467041,
          9.646331787109375,
          7.742466926574707,
          9.048988342285156,
          5.290090084075928,
          10.247977256774902,
          6.724141597747803,
          9.193451881408691,
          9.496269226074219,
          7.38286828994751,
          9.744629859924316,
          8.213132858276367,
          8.489629745483398,
          8.224211692810059,
          8.612765312194824,
          7.395656108856201,
          5.263019561767578,
          9.726327896118164,
          7.114065170288086,
          9.712484359741211,
          7.583319187164307,
          9.511140823364258,
          9.740355491638184,
          10.276066780090332,
          8.859769821166992,
          10.193078994750977,
          5.230263710021973,
          7.220943450927734,
          10.267227172851562,
          9.627487182617188,
          9.001677513122559,
          8.055542945861816,
          7.58953332901001,
          6.72620964050293,
          7.454516410827637,
          7.6788835525512695,
          8.176236152648926,
          10.657193183898926,
          7.448272228240967,
          10.607529640197754,
          9.748403549194336,
          7.568476676940918,
          10.340410232543945,
          13.907886505126953,
          8.740944862365723,
          8.671788215637207
         ]
        },
        {
         "hoverinfo": "text",
         "hovertext": [
          "Gérer les préférences de notification d'anniversaire/de consentement",
          "Afficher les fonctions supplémentaires de Time",
          "Afficher ma fiche de présence",
          "Gérer les exécutions de paie",
          "Afficher les anniversaires",
          "Afficher les rapports de congés",
          "Gérer les signalements de maladie et de blessure",
          "Gérer les politiques d’entreprise",
          "Afficher et gérer les avantages sociaux",
          "Afficher les équipes",
          "Afficher vos équipes",
          "Gérer le congé",
          "Gérer la rémunération de votre entreprise",
          "Modifier les détails de la date de naissance",
          "Gérer les congés de l'entreprise",
          "Afficher et gérer les avantages sociaux du collaborateur",
          "Afficher les modèles d'emploi",
          "Afficher un instantané remarquable pour Qi Cong Zhang",
          "Afficher vos tâches d’accueil et intégration",
          "Consulter votre paie",
          "Importer les valeurs des champs fluides",
          "Gestion des données des avantages sociaux",
          "Voir les résultats de la mesure de l'engagement",
          "Voir les unités d'entreprise",
          "Outils de base pour l’intégration",
          "Afficher les entités juridiques",
          "Afficher mon profil de collaborateur",
          "Gérer les formulaires d'avantages sociaux et les documents de régime",
          "Congés en vrac",
          "Afficher les horaires de l’équipe",
          "Afficher les familles d'emploi",
          "Afficher mes congés",
          "Gérer vos successeurs",
          "Afficher  l'organigramme de l'équipe",
          "Gérer les données de l'entreprise de paie",
          "Importer les données du travailleur",
          "Voir les nouvelles de l’entreprise",
          "Afficher les anniversaires de travail",
          "Consulter et demander un congé autorisé",
          "Inscription aux avantages sociaux complémentaires",
          "Afficher les postes",
          "Créer des critères",
          "Demander un congé",
          "Étalonner les attributs du talent",
          "Gérer les autorisations",
          "Terminer la mise au point",
          "Mise à jour des buts / objectifs",
          "Afficher votre rémunération",
          "Afficher mon calendrier",
          "Afficher vos avantages sociaux",
          "Ajouter une nouvelle embauche",
          "Afficher et approuver les fiches de présence de l’équipe",
          "Voir les classifications des professions",
          "Pointage à l'arrivée / au départ",
          "Gérer le calendrier de paie",
          "Afficher mon Org",
          "Voir l'emplacement des bureaux",
          "Afficher et approuver les congés de l’équipe",
          "Importer les données du collaborateur",
          "Afficher Wisely",
          "Afficher mes reconnaissances",
          "Cessation d'emploi en masse",
          null
         ],
         "marker": {
          "opacity": 0.5,
          "size": 5
         },
         "mode": "markers+text",
         "name": "1_les_afficher_grer",
         "text": [
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "1_les_afficher_grer"
         ],
         "textfont": {
          "size": 12
         },
         "type": "scattergl",
         "x": [
          5.510364532470703,
          5.929002285003662,
          6.259250640869141,
          6.541725158691406,
          5.600844860076904,
          6.236289024353027,
          6.53878116607666,
          6.431664943695068,
          5.816229343414307,
          6.146270751953125,
          6.304378509521484,
          6.398159980773926,
          6.464111804962158,
          6.577775478363037,
          6.433886528015137,
          5.813835620880127,
          5.5569281578063965,
          6.219635963439941,
          5.987354755401611,
          6.462048530578613,
          5.815256118774414,
          5.634487628936768,
          6.237320423126221,
          6.396665096282959,
          6.062536716461182,
          5.816566467285156,
          6.03828239440918,
          5.749380588531494,
          6.335677146911621,
          6.222420692443848,
          5.543763637542725,
          6.415196418762207,
          6.2626543045043945,
          6.267557621002197,
          6.502315998077393,
          5.755073547363281,
          6.461207866668701,
          5.669480800628662,
          5.999200344085693,
          5.718117713928223,
          5.923144817352295,
          5.3010406494140625,
          6.3610920906066895,
          5.934629917144775,
          5.888856887817383,
          5.772174835205078,
          5.3416619300842285,
          6.2643280029296875,
          6.5763936042785645,
          5.724579334259033,
          5.7676496505737305,
          6.245919227600098,
          6.029401779174805,
          5.778074741363525,
          6.60933780670166,
          6.438015460968018,
          6.199752330780029,
          6.307998180389404,
          5.860546112060547,
          6.450164318084717,
          6.481350898742676,
          5.557960033416748,
          6.079770088195801
         ],
         "y": [
          14.504573822021484,
          14.678389549255371,
          14.776219367980957,
          14.123250961303711,
          14.613862991333008,
          14.453850746154785,
          14.098565101623535,
          13.970718383789062,
          14.675719261169434,
          14.590503692626953,
          14.65478515625,
          14.283883094787598,
          13.739642143249512,
          14.22648811340332,
          13.952229499816895,
          14.567841529846191,
          14.931509017944336,
          14.629375457763672,
          14.700738906860352,
          13.892618179321289,
          14.141672134399414,
          14.408376693725586,
          14.170267105102539,
          13.928906440734863,
          14.708724021911621,
          14.626514434814453,
          14.49846076965332,
          14.551732063293457,
          14.438193321228027,
          14.540865898132324,
          14.936323165893555,
          14.748863220214844,
          13.99923038482666,
          14.59324836730957,
          13.971404075622559,
          14.130240440368652,
          13.918397903442383,
          14.543999671936035,
          14.044602394104004,
          14.50252628326416,
          14.698551177978516,
          14.023415565490723,
          14.348702430725098,
          14.075970649719238,
          14.109498023986816,
          14.807597160339355,
          13.369242668151855,
          13.75196361541748,
          14.631407737731934,
          14.592974662780762,
          14.60753059387207,
          14.660982131958008,
          13.919504165649414,
          14.915971755981445,
          14.178998947143555,
          14.765018463134766,
          14.388895034790039,
          14.501412391662598,
          14.09395980834961,
          14.79726505279541,
          14.817906379699707,
          14.938457489013672,
          14.394556999206543
         ]
        },
        {
         "hoverinfo": "text",
         "hovertext": [
          "Enviar correos electrónicos de registro",
          "Ver su pago",
          "Ver las noticias de la compañía",
          "Agregar mis contactos personales",
          "Agregar mi dirección postal",
          "Administrar datos de pago de colaboradores",
          "Panel de control de licencias y certificaciones",
          "Crear puesto",
          "Solución de problemas de gestión de usuarios",
          "Enviar un pulso de rendimiento",
          "Ver panel de control de los datos de entrada de RR. HH.",
          "Ajouter mon adresse postale",
          "Gestionar el consentimiento para los puntos de referencia de la ADP",
          "Solicitar tiempo libre",
          "Administrar el tiempo libre de la empresa",
          "Panel de control de registro",
          "Aller à BVCore",
          "Ver puestos",
          "Ver movimientos de las personas",
          "Declaración de remesa",
          "Ajouter mes coordonnées personnelles",
          "Agregar mi dirección física",
          "Gestionar usuarios",
          "Ver mis reconocimientos",
          "Acceso a cambio de producto",
          "Reconocer a alguien",
          "Configurar los recordatorios de caducidad de permiso de trabajo",
          "Informes y análisis",
          "Ver mis identificaciones",
          "Administración de datos de beneficios",
          "Rastreador de AOID",
          "Ver y gestionar beneficios del colaborador",
          "Ver Mis retenciones fiscales",
          "Ver mis licencias y certificaciones",
          "Statement Of Remittance",
          "Registros de empleo",
          null
         ],
         "marker": {
          "opacity": 0.5,
          "size": 5
         },
         "mode": "markers+text",
         "name": "2_de_ver_mis",
         "text": [
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "2_de_ver_mis"
         ],
         "textfont": {
          "size": 12
         },
         "type": "scattergl",
         "x": [
          4.243731498718262,
          4.7576727867126465,
          4.63575553894043,
          5.16911506652832,
          5.236477851867676,
          4.283945560455322,
          4.107620716094971,
          4.768316268920898,
          4.924204349517822,
          4.786270618438721,
          4.13972282409668,
          5.433409214019775,
          4.515289783477783,
          4.859224796295166,
          4.67449951171875,
          4.237851142883301,
          4.86868953704834,
          4.800090312957764,
          4.702214241027832,
          4.7143425941467285,
          5.419971942901611,
          5.015093803405762,
          4.94247579574585,
          4.649747371673584,
          4.63610315322876,
          4.803473472595215,
          4.110645294189453,
          4.322048664093018,
          4.700778484344482,
          4.162100315093994,
          4.8890485763549805,
          4.665257930755615,
          4.539511680603027,
          4.413403034210205,
          4.4560065269470215,
          4.5602874755859375,
          4.670677661895752
         ],
         "y": [
          12.886436462402344,
          12.22465705871582,
          12.329854011535645,
          12.529475212097168,
          12.583853721618652,
          12.526602745056152,
          12.562870979309082,
          12.361759185791016,
          12.56933307647705,
          12.346603393554688,
          12.809471130371094,
          12.656750679016113,
          12.806511878967285,
          12.66469669342041,
          12.594487190246582,
          12.913315773010254,
          12.092267990112305,
          12.243475914001465,
          12.40422248840332,
          12.061509132385254,
          12.60485553741455,
          12.621195793151855,
          12.502816200256348,
          12.037321090698242,
          12.723730087280273,
          11.925707817077637,
          12.488405227661133,
          12.328514099121094,
          12.16463851928711,
          12.561619758605957,
          12.656720161437988,
          12.29097843170166,
          11.94420051574707,
          12.186092376708984,
          11.837449073791504,
          12.814238548278809,
          12.440463066101074
         ]
        },
        {
         "hoverinfo": "text",
         "hovertext": [
          "View Payroll Dashboard",
          "eI-9 Dashboard",
          "Integration Core Tooling",
          "Azure AD Dashboard",
          "Global Integration Dashboard",
          "Garnishments Dashboard",
          "Integrations Health Monitor",
          "Document Management Dashboard",
          "IDX Azure Monitoring Dashboard",
          "Panel de control eI-9",
          "Execute Integration",
          "Outbound Integrations Execution Dashboard",
          "Datacloud Admin Dashboard",
          "Manage ACA Report Dashboard",
          "View HR Dashboard",
          "Integrations Dashboard",
          "Document Cloud",
          "Access ADPRM Recruiting Dashboard",
          "Track Applications",
          "Integration Dashboard",
          "Bulk Document Migration Dashboard",
          "Celergo Dashboard",
          "Apex Inbound Integration",
          "Special Accommodation Dashboard",
          "Manage eI-9 Dashboard",
          "Document Cloud Dashboard",
          "Approval Management Dashboard",
          "View Exit Interviews Dashboard",
          "Manage EV5 Integration",
          "View Cornerstone Employee Sync Dashboard",
          "Registration Dashboard",
          "Lightweight Integrations",
          "Hiring Dashboard",
          "View HR Inbound Data Dashboard",
          "DataCloud Dashboard",
          null
         ],
         "marker": {
          "opacity": 0.5,
          "size": 5
         },
         "mode": "markers+text",
         "name": "3_dashboard_integration_integrations",
         "text": [
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "3_dashboard_integration_integrations"
         ],
         "textfont": {
          "size": 12
         },
         "type": "scattergl",
         "x": [
          1.535766363143921,
          1.5539896488189697,
          1.348726749420166,
          1.593746304512024,
          1.3114205598831177,
          1.1664941310882568,
          1.299312949180603,
          1.263715147972107,
          1.5913747549057007,
          1.821203589439392,
          1.3402774333953857,
          1.3226618766784668,
          1.2824883460998535,
          1.2970616817474365,
          1.5751127004623413,
          1.3295550346374512,
          1.2307907342910767,
          1.5993260145187378,
          2.3304696083068848,
          1.363421082496643,
          1.1993249654769897,
          1.202931523323059,
          1.3243019580841064,
          1.4683289527893066,
          1.6507279872894287,
          1.1995638608932495,
          1.3137327432632446,
          1.4840823411941528,
          1.3363261222839355,
          1.5894981622695923,
          1.427470088005066,
          1.3339265584945679,
          1.5442477464675903,
          1.5558253526687622,
          1.4205408096313477,
          1.434506893157959
         ],
         "y": [
          11.861189842224121,
          12.662202835083008,
          13.028861045837402,
          12.298238754272461,
          12.693510055541992,
          12.548013687133789,
          12.822519302368164,
          12.491703987121582,
          12.390899658203125,
          12.800202369689941,
          13.054234504699707,
          12.883255004882812,
          12.458975791931152,
          12.326640129089355,
          11.943055152893066,
          12.602690696716309,
          12.480152130126953,
          12.083550453186035,
          12.46265983581543,
          12.65826416015625,
          12.535955429077148,
          12.572355270385742,
          13.034445762634277,
          12.51198959350586,
          12.696945190429688,
          12.47541332244873,
          12.426078796386719,
          12.031481742858887,
          12.816277503967285,
          11.90587043762207,
          12.361919403076172,
          13.072999000549316,
          12.075474739074707,
          11.982702255249023,
          12.40933895111084,
          12.498858451843262
         ]
        },
        {
         "hoverinfo": "text",
         "hovertext": [
          "tabaccesstestcase",
          "Tablero de retenciones",
          "Gérer le tableau de bord eI-9",
          "Tableau de bord eI-9",
          "Tableau de bord de l'exécution des intégrations sortantes",
          "Tableau de bord des enregistrements",
          "Tableau de bord des accommodements spéciaux",
          "Gérer le tableau de bord du rapport ACA",
          "Afficher le tableau de bord Paie",
          "Tableau de bord de gestion du document",
          "Tableau de bord DataCloud",
          "Afficher le tableau de bord des entrevues de départ",
          "Afficher tableau de bord données entrantes RH",
          "Tableau de bord Datacloud de l'administrateur",
          "Tableau de bord de gestion des approbations",
          "Afficher le tableau d'emplois",
          "Tableau de bord de migration des documents en masse",
          "Afficher le tableau de bord de synchronisation des employés de Cornerstone",
          "Tableau de bord de la surveillance IDX Azure",
          "Tableau de bord d'embauche",
          null
         ],
         "marker": {
          "opacity": 0.5,
          "size": 5
         },
         "mode": "markers+text",
         "name": "4_tableau_bord_de",
         "text": [
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "4_tableau_bord_de"
         ],
         "textfont": {
          "size": 12
         },
         "type": "scattergl",
         "x": [
          3.774259090423584,
          3.8403995037078857,
          3.8214762210845947,
          3.862924575805664,
          3.870774745941162,
          3.7856764793395996,
          3.8553948402404785,
          3.7960762977600098,
          3.8619728088378906,
          3.7736284732818604,
          3.3225910663604736,
          3.9208908081054688,
          3.8856630325317383,
          3.289916753768921,
          3.870248794555664,
          3.9099574089050293,
          3.644735813140869,
          3.654954671859741,
          3.164397716522217,
          3.858351230621338,
          3.7382144927978516
         ],
         "y": [
          14.480945587158203,
          14.39660930633545,
          14.499542236328125,
          14.52878475189209,
          14.287302017211914,
          14.131997108459473,
          14.284038543701172,
          14.512214660644531,
          14.471060752868652,
          14.415928840637207,
          14.030672073364258,
          14.35857105255127,
          14.406723976135254,
          13.97426986694336,
          14.37430477142334,
          14.52452278137207,
          14.308371543884277,
          13.955062866210938,
          13.882368087768555,
          14.449332237243652,
          14.313631057739258
         ]
        }
       ],
       "layout": {
        "annotations": [
         {
          "showarrow": false,
          "text": "D1",
          "x": -0.8590858936309814,
          "y": 10.8124751329422,
          "yshift": 10
         },
         {
          "showarrow": false,
          "text": "D2",
          "x": 3.370826292037964,
          "xshift": 10,
          "y": 17.179226112365722
         }
        ],
        "height": 750,
        "shapes": [
         {
          "line": {
           "color": "#CFD8DC",
           "width": 2
          },
          "type": "line",
          "x0": 3.370826292037964,
          "x1": 3.370826292037964,
          "y0": 4.445724153518677,
          "y1": 17.179226112365722
         },
         {
          "line": {
           "color": "#9E9E9E",
           "width": 2
          },
          "type": "line",
          "x0": -0.8590858936309814,
          "x1": 7.600738477706909,
          "y0": 10.8124751329422,
          "y1": 10.8124751329422
         }
        ],
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "rgb(36,36,36)"
            },
            "error_y": {
             "color": "rgb(36,36,36)"
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
             "endlinecolor": "rgb(36,36,36)",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "rgb(36,36,36)"
            },
            "baxis": {
             "endlinecolor": "rgb(36,36,36)",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "rgb(36,36,36)"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "colorscale": [
             [
              0,
              "#440154"
             ],
             [
              0.1111111111111111,
              "#482878"
             ],
             [
              0.2222222222222222,
              "#3e4989"
             ],
             [
              0.3333333333333333,
              "#31688e"
             ],
             [
              0.4444444444444444,
              "#26828e"
             ],
             [
              0.5555555555555556,
              "#1f9e89"
             ],
             [
              0.6666666666666666,
              "#35b779"
             ],
             [
              0.7777777777777778,
              "#6ece58"
             ],
             [
              0.8888888888888888,
              "#b5de2b"
             ],
             [
              1,
              "#fde725"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "colorscale": [
             [
              0,
              "#440154"
             ],
             [
              0.1111111111111111,
              "#482878"
             ],
             [
              0.2222222222222222,
              "#3e4989"
             ],
             [
              0.3333333333333333,
              "#31688e"
             ],
             [
              0.4444444444444444,
              "#26828e"
             ],
             [
              0.5555555555555556,
              "#1f9e89"
             ],
             [
              0.6666666666666666,
              "#35b779"
             ],
             [
              0.7777777777777778,
              "#6ece58"
             ],
             [
              0.8888888888888888,
              "#b5de2b"
             ],
             [
              1,
              "#fde725"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "colorscale": [
             [
              0,
              "#440154"
             ],
             [
              0.1111111111111111,
              "#482878"
             ],
             [
              0.2222222222222222,
              "#3e4989"
             ],
             [
              0.3333333333333333,
              "#31688e"
             ],
             [
              0.4444444444444444,
              "#26828e"
             ],
             [
              0.5555555555555556,
              "#1f9e89"
             ],
             [
              0.6666666666666666,
              "#35b779"
             ],
             [
              0.7777777777777778,
              "#6ece58"
             ],
             [
              0.8888888888888888,
              "#b5de2b"
             ],
             [
              1,
              "#fde725"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "line": {
              "color": "white",
              "width": 0.6
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "colorscale": [
             [
              0,
              "#440154"
             ],
             [
              0.1111111111111111,
              "#482878"
             ],
             [
              0.2222222222222222,
              "#3e4989"
             ],
             [
              0.3333333333333333,
              "#31688e"
             ],
             [
              0.4444444444444444,
              "#26828e"
             ],
             [
              0.5555555555555556,
              "#1f9e89"
             ],
             [
              0.6666666666666666,
              "#35b779"
             ],
             [
              0.7777777777777778,
              "#6ece58"
             ],
             [
              0.8888888888888888,
              "#b5de2b"
             ],
             [
              1,
              "#fde725"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "colorscale": [
             [
              0,
              "#440154"
             ],
             [
              0.1111111111111111,
              "#482878"
             ],
             [
              0.2222222222222222,
              "#3e4989"
             ],
             [
              0.3333333333333333,
              "#31688e"
             ],
             [
              0.4444444444444444,
              "#26828e"
             ],
             [
              0.5555555555555556,
              "#1f9e89"
             ],
             [
              0.6666666666666666,
              "#35b779"
             ],
             [
              0.7777777777777778,
              "#6ece58"
             ],
             [
              0.8888888888888888,
              "#b5de2b"
             ],
             [
              1,
              "#fde725"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
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
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 1,
              "tickcolor": "rgb(36,36,36)",
              "ticks": "outside"
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 1,
             "tickcolor": "rgb(36,36,36)",
             "ticks": "outside"
            },
            "colorscale": [
             [
              0,
              "#440154"
             ],
             [
              0.1111111111111111,
              "#482878"
             ],
             [
              0.2222222222222222,
              "#3e4989"
             ],
             [
              0.3333333333333333,
              "#31688e"
             ],
             [
              0.4444444444444444,
              "#26828e"
             ],
             [
              0.5555555555555556,
              "#1f9e89"
             ],
             [
              0.6666666666666666,
              "#35b779"
             ],
             [
              0.7777777777777778,
              "#6ece58"
             ],
             [
              0.8888888888888888,
              "#b5de2b"
             ],
             [
              1,
              "#fde725"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "rgb(237,237,237)"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "rgb(217,217,217)"
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
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 1,
            "tickcolor": "rgb(36,36,36)",
            "ticks": "outside"
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "rgb(103,0,31)"
            ],
            [
             0.1,
             "rgb(178,24,43)"
            ],
            [
             0.2,
             "rgb(214,96,77)"
            ],
            [
             0.3,
             "rgb(244,165,130)"
            ],
            [
             0.4,
             "rgb(253,219,199)"
            ],
            [
             0.5,
             "rgb(247,247,247)"
            ],
            [
             0.6,
             "rgb(209,229,240)"
            ],
            [
             0.7,
             "rgb(146,197,222)"
            ],
            [
             0.8,
             "rgb(67,147,195)"
            ],
            [
             0.9,
             "rgb(33,102,172)"
            ],
            [
             1,
             "rgb(5,48,97)"
            ]
           ],
           "sequential": [
            [
             0,
             "#440154"
            ],
            [
             0.1111111111111111,
             "#482878"
            ],
            [
             0.2222222222222222,
             "#3e4989"
            ],
            [
             0.3333333333333333,
             "#31688e"
            ],
            [
             0.4444444444444444,
             "#26828e"
            ],
            [
             0.5555555555555556,
             "#1f9e89"
            ],
            [
             0.6666666666666666,
             "#35b779"
            ],
            [
             0.7777777777777778,
             "#6ece58"
            ],
            [
             0.8888888888888888,
             "#b5de2b"
            ],
            [
             1,
             "#fde725"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#440154"
            ],
            [
             0.1111111111111111,
             "#482878"
            ],
            [
             0.2222222222222222,
             "#3e4989"
            ],
            [
             0.3333333333333333,
             "#31688e"
            ],
            [
             0.4444444444444444,
             "#26828e"
            ],
            [
             0.5555555555555556,
             "#1f9e89"
            ],
            [
             0.6666666666666666,
             "#35b779"
            ],
            [
             0.7777777777777778,
             "#6ece58"
            ],
            [
             0.8888888888888888,
             "#b5de2b"
            ],
            [
             1,
             "#fde725"
            ]
           ]
          },
          "colorway": [
           "#1F77B4",
           "#FF7F0E",
           "#2CA02C",
           "#D62728",
           "#9467BD",
           "#8C564B",
           "#E377C2",
           "#7F7F7F",
           "#BCBD22",
           "#17BECF"
          ],
          "font": {
           "color": "rgb(36,36,36)"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "white",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
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
            "gridcolor": "rgb(232,232,232)",
            "linecolor": "rgb(36,36,36)",
            "showgrid": false,
            "showline": true,
            "ticks": "outside"
           },
           "bgcolor": "white",
           "radialaxis": {
            "gridcolor": "rgb(232,232,232)",
            "linecolor": "rgb(36,36,36)",
            "showgrid": false,
            "showline": true,
            "ticks": "outside"
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "white",
            "gridcolor": "rgb(232,232,232)",
            "gridwidth": 2,
            "linecolor": "rgb(36,36,36)",
            "showbackground": true,
            "showgrid": false,
            "showline": true,
            "ticks": "outside",
            "zeroline": false,
            "zerolinecolor": "rgb(36,36,36)"
           },
           "yaxis": {
            "backgroundcolor": "white",
            "gridcolor": "rgb(232,232,232)",
            "gridwidth": 2,
            "linecolor": "rgb(36,36,36)",
            "showbackground": true,
            "showgrid": false,
            "showline": true,
            "ticks": "outside",
            "zeroline": false,
            "zerolinecolor": "rgb(36,36,36)"
           },
           "zaxis": {
            "backgroundcolor": "white",
            "gridcolor": "rgb(232,232,232)",
            "gridwidth": 2,
            "linecolor": "rgb(36,36,36)",
            "showbackground": true,
            "showgrid": false,
            "showline": true,
            "ticks": "outside",
            "zeroline": false,
            "zerolinecolor": "rgb(36,36,36)"
           }
          },
          "shapedefaults": {
           "fillcolor": "black",
           "line": {
            "width": 0
           },
           "opacity": 0.3
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "rgb(232,232,232)",
            "linecolor": "rgb(36,36,36)",
            "showgrid": false,
            "showline": true,
            "ticks": "outside"
           },
           "baxis": {
            "gridcolor": "rgb(232,232,232)",
            "linecolor": "rgb(36,36,36)",
            "showgrid": false,
            "showline": true,
            "ticks": "outside"
           },
           "bgcolor": "white",
           "caxis": {
            "gridcolor": "rgb(232,232,232)",
            "linecolor": "rgb(36,36,36)",
            "showgrid": false,
            "showline": true,
            "ticks": "outside"
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "rgb(232,232,232)",
           "linecolor": "rgb(36,36,36)",
           "showgrid": false,
           "showline": true,
           "ticks": "outside",
           "title": {
            "standoff": 15
           },
           "zeroline": false,
           "zerolinecolor": "rgb(36,36,36)"
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "rgb(232,232,232)",
           "linecolor": "rgb(36,36,36)",
           "showgrid": false,
           "showline": true,
           "ticks": "outside",
           "title": {
            "standoff": 15
           },
           "zeroline": false,
           "zerolinecolor": "rgb(36,36,36)"
          }
         }
        },
        "title": {
         "font": {
          "color": "Black",
          "size": 22
         },
         "text": "<b>Documents and Topics</b>",
         "x": 0.5,
         "xanchor": "center",
         "yanchor": "top"
        },
        "width": 1200,
        "xaxis": {
         "visible": false
        },
        "yaxis": {
         "visible": false
        }
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualize a selection of topics and documents\n",
    "topic_model.visualize_documents(caption, topics=[0, 1, 2, 3, 4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "6c0463f7",
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
         "marker": {
          "color": "#D55E00"
         },
         "orientation": "h",
         "type": "bar",
         "x": [
          0.034921086676854396,
          0.03929516603610824,
          0.040315697583632085,
          0.04090477351522098,
          0.04533393523156688,
          0.048066049188960604,
          0.06733083355317213,
          0.0787453906525927,
          0.1045911326793215,
          0.1298164988678768
         ],
         "xaxis": "x",
         "y": [
          "team  ",
          "data  ",
          "your  ",
          "and  ",
          "import  ",
          "management  ",
          "configure  ",
          "my  ",
          "manage  ",
          "view  "
         ],
         "yaxis": "y"
        },
        {
         "marker": {
          "color": "#0072B2"
         },
         "orientation": "h",
         "type": "bar",
         "x": [
          0.06470458284238813,
          0.06596455789979146,
          0.06767677499841497,
          0.06767677499841497,
          0.07764549941086574,
          0.09606087965775054,
          0.11863750024365267,
          0.1371033978775642,
          0.1996201902530855,
          0.23658689215180503
         ],
         "xaxis": "x2",
         "y": [
          "congs  ",
          "des  ",
          "avantages  ",
          "voir  ",
          "sociaux  ",
          "et  ",
          "de  ",
          "grer  ",
          "afficher  ",
          "les  "
         ],
         "yaxis": "y2"
        },
        {
         "marker": {
          "color": "#CC79A7"
         },
         "orientation": "h",
         "type": "bar",
         "x": [
          0.06521494861805935,
          0.06977234730314436,
          0.07591568673735244,
          0.08001015368752504,
          0.08530698472546903,
          0.08530698472546903,
          0.08530698472546903,
          0.12652614456225406,
          0.20013687049446433,
          0.24739291438768377
         ],
         "xaxis": "x3",
         "y": [
          "la  ",
          "control  ",
          "panel  ",
          "los  ",
          "gestionar  ",
          "datos  ",
          "agregar  ",
          "mis  ",
          "ver  ",
          "de  "
         ],
         "yaxis": "y3"
        },
        {
         "marker": {
          "color": "#E69F00"
         },
         "orientation": "h",
         "type": "bar",
         "x": [
          0.08257458169033033,
          0.08804118295384947,
          0.08804118295384947,
          0.08804118295384947,
          0.09577261705022207,
          0.11752332273763215,
          0.1566977636501762,
          0.16514916338066066,
          0.22472399348721722,
          0.6066743159255191
         ],
         "xaxis": "x4",
         "y": [
          "datacloud  ",
          "azure  ",
          "inbound  ",
          "hr  ",
          "cloud  ",
          "ei9  ",
          "document  ",
          "integrations  ",
          "integration  ",
          "dashboard  "
         ],
         "yaxis": "y4"
        },
        {
         "marker": {
          "color": "#56B4E9"
         },
         "orientation": "h",
         "type": "bar",
         "x": [
          0.05690843754746643,
          0.06624620902283872,
          0.06624620902283872,
          0.06981915850239313,
          0.09046322984971203,
          0.16142546282794495,
          0.17072531264239926,
          0.30149060803007044,
          0.3994822754815837,
          0.41509404727185845
         ],
         "xaxis": "x5",
         "y": [
          "du  ",
          "ei9  ",
          "gestion  ",
          "datacloud  ",
          "afficher  ",
          "des  ",
          "le  ",
          "de  ",
          "bord  ",
          "tableau  "
         ],
         "yaxis": "y5"
        }
       ],
       "layout": {
        "annotations": [
         {
          "font": {
           "size": 16
          },
          "showarrow": false,
          "text": "Topic 0",
          "x": 0.0875,
          "xanchor": "center",
          "xref": "paper",
          "y": 1,
          "yanchor": "bottom",
          "yref": "paper"
         },
         {
          "font": {
           "size": 16
          },
          "showarrow": false,
          "text": "Topic 1",
          "x": 0.36250000000000004,
          "xanchor": "center",
          "xref": "paper",
          "y": 1,
          "yanchor": "bottom",
          "yref": "paper"
         },
         {
          "font": {
           "size": 16
          },
          "showarrow": false,
          "text": "Topic 2",
          "x": 0.6375000000000001,
          "xanchor": "center",
          "xref": "paper",
          "y": 1,
          "yanchor": "bottom",
          "yref": "paper"
         },
         {
          "font": {
           "size": 16
          },
          "showarrow": false,
          "text": "Topic 3",
          "x": 0.9125,
          "xanchor": "center",
          "xref": "paper",
          "y": 1,
          "yanchor": "bottom",
          "yref": "paper"
         },
         {
          "font": {
           "size": 16
          },
          "showarrow": false,
          "text": "Topic 4",
          "x": 0.0875,
          "xanchor": "center",
          "xref": "paper",
          "y": 0.4,
          "yanchor": "bottom",
          "yref": "paper"
         }
        ],
        "height": 500,
        "hoverlabel": {
         "bgcolor": "white",
         "font": {
          "family": "Rockwell",
          "size": 16
         }
        },
        "showlegend": false,
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
        "title": {
         "font": {
          "color": "Black",
          "size": 22
         },
         "text": "Topic Word Scores",
         "x": 0.5,
         "xanchor": "center",
         "yanchor": "top"
        },
        "width": 1000,
        "xaxis": {
         "anchor": "y",
         "domain": [
          0,
          0.175
         ],
         "showgrid": true
        },
        "xaxis2": {
         "anchor": "y2",
         "domain": [
          0.275,
          0.45
         ],
         "showgrid": true
        },
        "xaxis3": {
         "anchor": "y3",
         "domain": [
          0.55,
          0.7250000000000001
         ],
         "showgrid": true
        },
        "xaxis4": {
         "anchor": "y4",
         "domain": [
          0.825,
          1
         ],
         "showgrid": true
        },
        "xaxis5": {
         "anchor": "y5",
         "domain": [
          0,
          0.175
         ],
         "showgrid": true
        },
        "xaxis6": {
         "anchor": "y6",
         "domain": [
          0.275,
          0.45
         ],
         "showgrid": true
        },
        "xaxis7": {
         "anchor": "y7",
         "domain": [
          0.55,
          0.7250000000000001
         ],
         "showgrid": true
        },
        "xaxis8": {
         "anchor": "y8",
         "domain": [
          0.825,
          1
         ],
         "showgrid": true
        },
        "yaxis": {
         "anchor": "x",
         "domain": [
          0.6000000000000001,
          1
         ],
         "showgrid": true
        },
        "yaxis2": {
         "anchor": "x2",
         "domain": [
          0.6000000000000001,
          1
         ],
         "showgrid": true
        },
        "yaxis3": {
         "anchor": "x3",
         "domain": [
          0.6000000000000001,
          1
         ],
         "showgrid": true
        },
        "yaxis4": {
         "anchor": "x4",
         "domain": [
          0.6000000000000001,
          1
         ],
         "showgrid": true
        },
        "yaxis5": {
         "anchor": "x5",
         "domain": [
          0,
          0.4
         ],
         "showgrid": true
        },
        "yaxis6": {
         "anchor": "x6",
         "domain": [
          0,
          0.4
         ],
         "showgrid": true
        },
        "yaxis7": {
         "anchor": "x7",
         "domain": [
          0,
          0.4
         ],
         "showgrid": true
        },
        "yaxis8": {
         "anchor": "x8",
         "domain": [
          0,
          0.4
         ],
         "showgrid": true
        }
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "topic_model.visualize_barchart(topics=list([0, 1, 2, 3, 4]), n_words=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "322674bf",
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
         "hoverinfo": "text",
         "marker": {
          "color": "rgb(61,153,112)"
         },
         "mode": "lines",
         "type": "scatter",
         "x": [
          0,
          0.7705917557702742,
          0.7705917557702742,
          0
         ],
         "xaxis": "x",
         "y": [
          -5,
          -5,
          -15,
          -15
         ],
         "yaxis": "y"
        },
        {
         "hoverinfo": "text",
         "marker": {
          "color": "rgb(61,153,112)"
         },
         "mode": "lines",
         "type": "scatter",
         "x": [
          0.7705917557702742,
          0.8581809601243204,
          0.8581809601243204,
          0
         ],
         "xaxis": "x",
         "y": [
          -10,
          -10,
          -25,
          -25
         ],
         "yaxis": "y"
        },
        {
         "hoverinfo": "text",
         "marker": {
          "color": "rgb(255,65,54)"
         },
         "mode": "lines",
         "type": "scatter",
         "x": [
          0,
          0.8983364314373581,
          0.8983364314373581,
          0
         ],
         "xaxis": "x",
         "y": [
          -35,
          -35,
          -45,
          -45
         ],
         "yaxis": "y"
        },
        {
         "hoverinfo": "text",
         "marker": {
          "color": "rgb(0,116,217)"
         },
         "mode": "lines",
         "type": "scatter",
         "x": [
          0.8581809601243204,
          1.13216169805145,
          1.13216169805145,
          0.8983364314373581
         ],
         "xaxis": "x",
         "y": [
          -17.5,
          -17.5,
          -40,
          -40
         ],
         "yaxis": "y"
        }
       ],
       "layout": {
        "autosize": false,
        "height": 275,
        "hoverlabel": {
         "bgcolor": "white",
         "font": {
          "family": "Rockwell",
          "size": 16
         }
        },
        "hovermode": "closest",
        "plot_bgcolor": "#ECEFF1",
        "showlegend": false,
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
        "title": {
         "font": {
          "color": "Black",
          "size": 22
         },
         "text": "<b>Hierarchical Clustering</b>",
         "x": 0.5,
         "xanchor": "center",
         "yanchor": "top"
        },
        "width": 1000,
        "xaxis": {
         "mirror": "allticks",
         "rangemode": "tozero",
         "showgrid": false,
         "showline": true,
         "showticklabels": true,
         "ticks": "outside",
         "type": "linear",
         "zeroline": false
        },
        "yaxis": {
         "mirror": "allticks",
         "range": [
          -50,
          0
         ],
         "rangemode": "tozero",
         "showgrid": false,
         "showline": true,
         "showticklabels": true,
         "tickmode": "array",
         "ticks": "outside",
         "ticktext": [
          "1_les_afficher_grer",
          "4_tableau_bord_de",
          "2_de_ver_mis",
          "0_view_manage_my",
          "3_dashboard_integration_int..."
         ],
         "tickvals": [
          -5,
          -15,
          -25,
          -35,
          -45
         ],
         "type": "linear",
         "zeroline": false
        }
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualize barchart with ranked keywords\n",
    "topic_model.visualize_barchart()\n",
    "\n",
    "# Visualize relationships between topics\n",
    "topic_model.visualize_heatmap(n_clusters=4)\n",
    "\n",
    "# Visualize the potential hierarchical structure of topics\n",
    "topic_model.visualize_hierarchy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "0b0a514e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save original representations\n",
    "from copy import deepcopy\n",
    "original_topics = deepcopy(topic_model.topic_representations_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "46b3a90e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def topic_differences(model, original_topics, nr_topics=5):\n",
    "    \"\"\"Show the differences in topic representations between two models \"\"\"\n",
    "    df = pd.DataFrame(columns=[\"Topic\", \"Original\", \"Updated\"])\n",
    "    for topic in range(nr_topics):\n",
    "\n",
    "        # Extract top 5 words per topic per model\n",
    "        og_words = \" | \".join(list(zip(*original_topics[topic]))[0][:5])\n",
    "        new_words = \" | \".join(list(zip(*model.get_topic(topic)))[0][:5])\n",
    "        df.loc[len(df)] = [topic, og_words, new_words]\n",
    "    \n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "26566001",
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
       "      <th>Topic</th>\n",
       "      <th>Original</th>\n",
       "      <th>Updated</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>view | manage | my | configure | management</td>\n",
       "      <td>view | snapshot | benefits | access | reports</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>les | afficher | grer | de | et</td>\n",
       "      <td>une | votre | danniversairede | embauche | aff...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>de | ver | mis | datos | agregar</td>\n",
       "      <td>responsabilidades | reconocimientos | referenc...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>dashboard | integration | integrations | docum...</td>\n",
       "      <td>dashboard | integration | integrations | admin...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>tableau | bord | de | le | des</td>\n",
       "      <td>tableau | de | tablero | des | en</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Topic                                           Original  \\\n",
       "0      0        view | manage | my | configure | management   \n",
       "1      1                    les | afficher | grer | de | et   \n",
       "2      2                   de | ver | mis | datos | agregar   \n",
       "3      3  dashboard | integration | integrations | docum...   \n",
       "4      4                     tableau | bord | de | le | des   \n",
       "\n",
       "                                             Updated  \n",
       "0      view | snapshot | benefits | access | reports  \n",
       "1  une | votre | danniversairede | embauche | aff...  \n",
       "2  responsabilidades | reconocimientos | referenc...  \n",
       "3  dashboard | integration | integrations | admin...  \n",
       "4                  tableau | de | tablero | des | en  "
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from bertopic.representation import KeyBERTInspired\n",
    "\n",
    "# Update our topic representations using KeyBERTInspired\n",
    "representation_model = KeyBERTInspired()\n",
    "topic_model.update_topics(caption, representation_model=representation_model)\n",
    "\n",
    "# Show topic differences\n",
    "topic_differences(topic_model, original_topics)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "0b6afc57",
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
       "      <th>Topic</th>\n",
       "      <th>Original</th>\n",
       "      <th>Updated</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>view | manage | my | configure | management</td>\n",
       "      <td>manage | import | data | associate | configura...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>les | afficher | grer | de | et</td>\n",
       "      <td>congs | sociaux | avantages | lquipe | collabo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>de | ver | mis | datos | agregar</td>\n",
       "      <td>mis | datos | administrar | registro | adresse</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>dashboard | integration | integrations | docum...</td>\n",
       "      <td>dashboard | integrations | ei9 | cloud | hr</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>tableau | bord | de | le | des</td>\n",
       "      <td>des | datacloud | tablero | accommodements | l...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Topic                                           Original  \\\n",
       "0      0        view | manage | my | configure | management   \n",
       "1      1                    les | afficher | grer | de | et   \n",
       "2      2                   de | ver | mis | datos | agregar   \n",
       "3      3  dashboard | integration | integrations | docum...   \n",
       "4      4                     tableau | bord | de | le | des   \n",
       "\n",
       "                                             Updated  \n",
       "0  manage | import | data | associate | configura...  \n",
       "1  congs | sociaux | avantages | lquipe | collabo...  \n",
       "2     mis | datos | administrar | registro | adresse  \n",
       "3        dashboard | integrations | ei9 | cloud | hr  \n",
       "4  des | datacloud | tablero | accommodements | l...  "
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from bertopic.representation import MaximalMarginalRelevance\n",
    "\n",
    "# Update our topic representations to MaximalMarginalRelevance\n",
    "representation_model = MaximalMarginalRelevance(diversity=0.2)\n",
    "topic_model.update_topics(caption, representation_model=representation_model)\n",
    "\n",
    "# Show topic differences\n",
    "topic_differences(topic_model, original_topics)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "45f791a6",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Hardware accelerator e.g. GPU is available in the environment, but no `device` argument is passed to the `Pipeline` object. Model will be on CPU.\n"
     ]
    },
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
       "      <th>Topic</th>\n",
       "      <th>Original</th>\n",
       "      <th>Updated</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>view | manage | my | configure | management</td>\n",
       "      <td>Business |  |  |  |</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>les | afficher | grer | de | et</td>\n",
       "      <td>Business |  |  |  |</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>de | ver | mis | datos | agregar</td>\n",
       "      <td>Data protection |  |  |  |</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>dashboard | integration | integrations | docum...</td>\n",
       "      <td>integrations |  |  |  |</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>tableau | bord | de | le | des</td>\n",
       "      <td>Tableau de bord |  |  |  |</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Topic                                           Original  \\\n",
       "0      0        view | manage | my | configure | management   \n",
       "1      1                    les | afficher | grer | de | et   \n",
       "2      2                   de | ver | mis | datos | agregar   \n",
       "3      3  dashboard | integration | integrations | docum...   \n",
       "4      4                     tableau | bord | de | le | des   \n",
       "\n",
       "                       Updated  \n",
       "0         Business |  |  |  |   \n",
       "1         Business |  |  |  |   \n",
       "2  Data protection |  |  |  |   \n",
       "3     integrations |  |  |  |   \n",
       "4  Tableau de bord |  |  |  |   "
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from transformers import pipeline\n",
    "from bertopic.representation import TextGeneration\n",
    "\n",
    "prompt = \"\"\"I have a topic that contains the following documents: \n",
    "[DOCUMENTS]\n",
    "\n",
    "The topic is described by the following keywords: '[KEYWORDS]'.\n",
    "\n",
    "Based on the documents and keywords, what is this topic about?\"\"\"\n",
    "\n",
    "# Update our topic representations using Flan-T5\n",
    "generator = pipeline(\"text2text-generation\", model=\"google/flan-t5-small\")\n",
    "representation_model = TextGeneration(\n",
    "    generator, prompt=prompt, doc_length=50, tokenizer=\"whitespace\"\n",
    ")\n",
    "topic_model.update_topics(caption, representation_model=representation_model)\n",
    "\n",
    "# Show topic differences\n",
    "topic_differences(topic_model, original_topics)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "132ceb38",
   "metadata": {},
   "source": [
    "### Action Data Processing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "e21c0dbf",
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
   "cell_type": "code",
   "execution_count": 74,
   "id": "8d959319",
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
       "      <th>details_search_value</th>\n",
       "      <th>docId</th>\n",
       "      <th>caption</th>\n",
       "      <th>category</th>\n",
       "      <th>subtitle</th>\n",
       "      <th>resPos</th>\n",
       "      <th>locale</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>191428</th>\n",
       "      <td>bob</td>\n",
       "      <td>002!92e3eae6ab5e4f8cab69c0c0b87eb7c1!global!fr-CA</td>\n",
       "      <td>Gérer les exécutions de paie</td>\n",
       "      <td>actions</td>\n",
       "      <td>Traiter les données de paie au sein d'un group...</td>\n",
       "      <td>49.0</td>\n",
       "      <td>fr-CA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>191429</th>\n",
       "      <td>bob</td>\n",
       "      <td>002!e49f4d0ab6944efe8fe91d0e009e8546!global!fr-CA</td>\n",
       "      <td>Survey(s) - Voice of the Employee</td>\n",
       "      <td>actions</td>\n",
       "      <td>Survey(s) - Voice of the Employee</td>\n",
       "      <td>48.0</td>\n",
       "      <td>fr-CA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>191430</th>\n",
       "      <td>bob</td>\n",
       "      <td>002!0429ea12a5974851ab47620c9d7205c9!global!fr-CA</td>\n",
       "      <td>Afficher le centre de contrôle des critères</td>\n",
       "      <td>actions</td>\n",
       "      <td>Configurez et gérez les requêtes de recherche ...</td>\n",
       "      <td>47.0</td>\n",
       "      <td>fr-CA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>191431</th>\n",
       "      <td>bob</td>\n",
       "      <td>002!411afb6ee5534505a3da0498a526c0c7!global!fr-CA</td>\n",
       "      <td>Demander un congé</td>\n",
       "      <td>actions</td>\n",
       "      <td>Demander un congé</td>\n",
       "      <td>46.0</td>\n",
       "      <td>fr-CA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>191432</th>\n",
       "      <td>bob</td>\n",
       "      <td>002!f1a4a530f0a342438dbf690244e71208!global!fr-CA</td>\n",
       "      <td>Gérer les congés de l'entreprise</td>\n",
       "      <td>actions</td>\n",
       "      <td>Gérez les soldes des congés, les politiques, l...</td>\n",
       "      <td>45.0</td>\n",
       "      <td>fr-CA</td>\n",
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
       "    </tr>\n",
       "    <tr>\n",
       "      <th>502102</th>\n",
       "      <td>view</td>\n",
       "      <td>002!59a20db646fe42d6b86864fe0ff504e3!global!fr-CA</td>\n",
       "      <td>View/Edit or Approve Team Time Card</td>\n",
       "      <td>actions</td>\n",
       "      <td>Allows user to manage time cards for direct re...</td>\n",
       "      <td>4.0</td>\n",
       "      <td>fr-CA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>502103</th>\n",
       "      <td>view</td>\n",
       "      <td>002!1bc35add73214eff907471fcf172636c!global!fr-CA</td>\n",
       "      <td>View Client Provided Translations</td>\n",
       "      <td>actions</td>\n",
       "      <td>View Client Provided Translations</td>\n",
       "      <td>3.0</td>\n",
       "      <td>fr-CA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>502104</th>\n",
       "      <td>view</td>\n",
       "      <td>002!dc8ada1ef309484eb57f51dd9e13f003!global!fr-CA</td>\n",
       "      <td>View Client Administrators</td>\n",
       "      <td>actions</td>\n",
       "      <td>View Administrators</td>\n",
       "      <td>2.0</td>\n",
       "      <td>fr-CA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>502105</th>\n",
       "      <td>view</td>\n",
       "      <td>002!b25e3f5cdb624e54a6fbba255b1baa6a!global!fr-CA</td>\n",
       "      <td>View Client Management</td>\n",
       "      <td>actions</td>\n",
       "      <td>All client, your Client and archived client data</td>\n",
       "      <td>1.0</td>\n",
       "      <td>fr-CA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>502106</th>\n",
       "      <td>view</td>\n",
       "      <td>002!a7e85827908a4619b7eadec3a0331d10!global!fr-CA</td>\n",
       "      <td>View HR Dashboard</td>\n",
       "      <td>actions</td>\n",
       "      <td>Tableau de bord de RH</td>\n",
       "      <td>0.0</td>\n",
       "      <td>fr-CA</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       details_search_value  \\\n",
       "191428                  bob   \n",
       "191429                  bob   \n",
       "191430                  bob   \n",
       "191431                  bob   \n",
       "191432                  bob   \n",
       "...                     ...   \n",
       "502102                view    \n",
       "502103                view    \n",
       "502104                view    \n",
       "502105                view    \n",
       "502106                view    \n",
       "\n",
       "                                                    docId  \\\n",
       "191428  002!92e3eae6ab5e4f8cab69c0c0b87eb7c1!global!fr-CA   \n",
       "191429  002!e49f4d0ab6944efe8fe91d0e009e8546!global!fr-CA   \n",
       "191430  002!0429ea12a5974851ab47620c9d7205c9!global!fr-CA   \n",
       "191431  002!411afb6ee5534505a3da0498a526c0c7!global!fr-CA   \n",
       "191432  002!f1a4a530f0a342438dbf690244e71208!global!fr-CA   \n",
       "...                                                   ...   \n",
       "502102  002!59a20db646fe42d6b86864fe0ff504e3!global!fr-CA   \n",
       "502103  002!1bc35add73214eff907471fcf172636c!global!fr-CA   \n",
       "502104  002!dc8ada1ef309484eb57f51dd9e13f003!global!fr-CA   \n",
       "502105  002!b25e3f5cdb624e54a6fbba255b1baa6a!global!fr-CA   \n",
       "502106  002!a7e85827908a4619b7eadec3a0331d10!global!fr-CA   \n",
       "\n",
       "                                            caption category  \\\n",
       "191428                 Gérer les exécutions de paie  actions   \n",
       "191429            Survey(s) - Voice of the Employee  actions   \n",
       "191430  Afficher le centre de contrôle des critères  actions   \n",
       "191431                            Demander un congé  actions   \n",
       "191432             Gérer les congés de l'entreprise  actions   \n",
       "...                                             ...      ...   \n",
       "502102          View/Edit or Approve Team Time Card  actions   \n",
       "502103            View Client Provided Translations  actions   \n",
       "502104                   View Client Administrators  actions   \n",
       "502105                       View Client Management  actions   \n",
       "502106                            View HR Dashboard  actions   \n",
       "\n",
       "                                                 subtitle  resPos locale  \n",
       "191428  Traiter les données de paie au sein d'un group...    49.0  fr-CA  \n",
       "191429                  Survey(s) - Voice of the Employee    48.0  fr-CA  \n",
       "191430  Configurez et gérez les requêtes de recherche ...    47.0  fr-CA  \n",
       "191431                                  Demander un congé    46.0  fr-CA  \n",
       "191432  Gérez les soldes des congés, les politiques, l...    45.0  fr-CA  \n",
       "...                                                   ...     ...    ...  \n",
       "502102  Allows user to manage time cards for direct re...     4.0  fr-CA  \n",
       "502103                  View Client Provided Translations     3.0  fr-CA  \n",
       "502104                                View Administrators     2.0  fr-CA  \n",
       "502105   All client, your Client and archived client data     1.0  fr-CA  \n",
       "502106                              Tableau de bord de RH     0.0  fr-CA  \n",
       "\n",
       "[150 rows x 7 columns]"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fr_action = actions[actions['locale'].isin(['fr-CA'])]\n",
    "# res = filtered_actions.groupby(['caption', 'details_search_value']).size().sort_values(ascending=False)\n",
    "# res.groupby(['details_search_value']).size().sort_values(ascending=False)\n",
    "fr_action"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5f6aec23",
   "metadata": {},
   "outputs": [],
   "source": [
    "### Filter query text\n",
    "\n",
    "filtered_actions = actions[~actions['locale'].isin(['fr-CA'])]\n",
    "filtered_actions.groupby(['details_search_value']).size().sort_values(ascending=False)\n",
    "\n",
    "query_chr_num = filtered_actions[filtered_actions['details_search_value'].str.contains(r'(?=.*[a-zA-Z])(?=.*[0-9])')]\n",
    "query_list = list(set(query_chr_num['details_search_value']))\n",
    "filtered_actions_df = filtered_actions[filtered_actions['details_search_value'].isin(query_list)]\n",
    "filtered_actions_df"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "539defa3",
   "metadata": {},
   "source": [
    "### Need Action Scores to Compute Similarity Scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "198159d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "import numpy as np\n",
    "import os\n",
    "import csv\n",
    "import logging\n",
    "from sklearn.metrics.pairwise import paired_cosine_distances, paired_manhattan_distances, paired_euclidean_distances\n",
    "from scipy.stats import pearsonr, spearmanr\n",
    "from typing import List, Dict, Optional\n",
    "\n",
    "logger = logging.getLogger(__name__)  # Create a logger object\n",
    "\n",
    "def get_embedding(text: str, model_name: str = \"/Users/huanglin/Bitbucket/xinpeng-multilingual-raw-data/paraphrase-multilingual-MiniLM-L12-v2\") -> List[float]:\n",
    "    # Load model\n",
    "    model = SentenceTransformer(model_name)\n",
    "    model.max_seq_length = 384\n",
    "    \"\"\"Get embeddings from OpenAI API\"\"\"\n",
    "    return model.encode(text)\n",
    "\n",
    "class CaptionEmbeddingSimilarityEvaluator:\n",
    "    \n",
    "    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float], \n",
    "                 batch_size: int = 16, main_similarity: str = 'COSINE', name: str = \"\", \n",
    "                 show_progress_bar: bool = False, write_csv: bool = True, \n",
    "                 precision: Optional[str] = None, truncate_dim: Optional[int] = None):\n",
    "        \"\"\"Initialize evaluator with sentences, scores, and other settings.\"\"\"\n",
    "        self.sentences1 = sentences1\n",
    "        self.sentences2 = sentences2\n",
    "        self.scores = scores\n",
    "        self.batch_size = batch_size\n",
    "        self.main_similarity = main_similarity\n",
    "        self.name = name\n",
    "        self.show_progress_bar = show_progress_bar\n",
    "        self.write_csv = write_csv\n",
    "        self.precision = precision\n",
    "        self.truncate_dim = truncate_dim\n",
    "        self.csv_file = f\"similarity_evaluation_{name}.csv\" if name else \"similarity_evaluation_results.csv\"\n",
    "        self.csv_headers = [\"epoch\", \"steps\", \"cosine_pearson\", \"cosine_spearman\",\n",
    "                            \"euclidean_pearson\", \"euclidean_spearman\", \"manhattan_pearson\", \n",
    "                            \"manhattan_spearman\", \"dot_pearson\", \"dot_spearman\"]\n",
    "\n",
    "        assert len(sentences1) == len(sentences2) == len(scores), \"Sentence lists and scores must be of equal length.\"\n",
    "\n",
    "    def __call__(self, output_path: Optional[str] = None, epoch: int = -1, steps: int = -1) -> Dict[str, float]:\n",
    "        \"\"\"Evaluate the similarity between the two sentence sets.\"\"\"\n",
    "        \n",
    "        # Optional logging for the epoch and step context\n",
    "        if epoch != -1:\n",
    "            out_txt = f\" after epoch {epoch}\" if steps == -1 else f\" in epoch {epoch} after {steps} steps\"\n",
    "\n",
    "        # Get embeddings for sentences1 and sentences2\n",
    "        embeddings1 = np.array([get_embedding(sentence, model='text-embedding-3-small') for sentence in self.sentences1])\n",
    "        embeddings2 = np.array([get_embedding(sentence, model='text-embedding-3-small') for sentence in self.sentences2])\n",
    "\n",
    "        # Handle precision adjustments (e.g., binary precision)\n",
    "        if self.precision in (\"binary\", \"ubinary\"):\n",
    "            embeddings1 = np.unpackbits(embeddings1, axis=1)\n",
    "            embeddings2 = np.unpackbits(embeddings2, axis=1)\n",
    "\n",
    "        # Compute distances and similarity metrics\n",
    "        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)\n",
    "        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)\n",
    "        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)\n",
    "        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]\n",
    "\n",
    "        # Ground truth labels for the sentence pairs\n",
    "        labels = np.array(self.scores)\n",
    "\n",
    "        # Evaluate Pearson and Spearman correlations\n",
    "        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)\n",
    "        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)\n",
    "        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)\n",
    "        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)\n",
    "        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)\n",
    "        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)\n",
    "        eval_pearson_dot, _ = pearsonr(labels, dot_products)\n",
    "        eval_spearman_dot, _ = spearmanr(labels, dot_products)\n",
    "\n",
    "        # Log results\n",
    "        logger.info(f\"Cosine-Similarity :\\tPearson: {eval_pearson_cosine:.4f}\\tSpearman: {eval_spearman_cosine:.4f}\")\n",
    "        logger.info(f\"Manhattan-Distance:\\tPearson: {eval_pearson_manhattan:.4f}\\tSpearman: {eval_spearman_manhattan:.4f}\")\n",
    "        logger.info(f\"Euclidean-Distance:\\tPearson: {eval_pearson_euclidean:.4f}\\tSpearman: {eval_spearman_euclidean:.4f}\")\n",
    "        logger.info(f\"Dot-Product-Similarity:\\tPearson: {eval_pearson_dot:.4f}\\tSpearman: {eval_spearman_dot:.4f}\")\n",
    "\n",
    "        # Write results to CSV if needed\n",
    "        if output_path and self.write_csv:\n",
    "            csv_path = os.path.join(output_path, self.csv_file)\n",
    "            output_file_exists = os.path.isfile(csv_path)\n",
    "            with open(csv_path, mode=\"a\" if output_file_exists else \"w\", newline=\"\", encoding=\"utf-8\") as f:\n",
    "                writer = csv.writer(f)\n",
    "                if not output_file_exists:\n",
    "                    writer.writerow(self.csv_headers)\n",
    "                writer.writerow([epoch, steps, eval_pearson_cosine, eval_spearman_cosine,\n",
    "                                 eval_pearson_euclidean, eval_spearman_euclidean,\n",
    "                                 eval_pearson_manhattan, eval_spearman_manhattan,\n",
    "                                 eval_pearson_dot, eval_spearman_dot])\n",
    "\n",
    "        # Return a dictionary of the evaluation metrics\n",
    "        metrics = {\n",
    "            \"pearson_cosine\": eval_pearson_cosine, \"spearman_cosine\": eval_spearman_cosine,\n",
    "            \"pearson_manhattan\": eval_pearson_manhattan, \"spearman_manhattan\": eval_spearman_manhattan,\n",
    "            \"pearson_euclidean\": eval_pearson_euclidean, \"spearman_euclidean\": eval_spearman_euclidean,\n",
    "            \"pearson_dot\": eval_pearson_dot, \"spearman_dot\": eval_spearman_dot,\n",
    "            \"pearson_max\": max(eval_pearson_cosine, eval_pearson_manhattan, eval_pearson_euclidean, eval_pearson_dot),\n",
    "            \"spearman_max\": max(eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot),\n",
    "        }\n",
    "\n",
    "        return metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "34532b4b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Disciplinary Incidents',\n",
       " 'Ver panel de control de los datos de entrada de RR.\\xa0HH.',\n",
       " 'View Job Chart',\n",
       " 'Approve Time',\n",
       " 'Configurar los tipos de relaciones laborales',\n",
       " \"Afficher  l'organigramme de l'équipe\",\n",
       " 'View Scope Management',\n",
       " 'View Your Benefits',\n",
       " 'TestLinkWithSequence3',\n",
       " 'Ver mis identificaciones',\n",
       " 'Gestion des attributions',\n",
       " 'View Wisely',\n",
       " 'View Job Templates',\n",
       " 'View Birthdays',\n",
       " 'eI-9 Dashboard',\n",
       " 'Document Cloud Dashboard',\n",
       " 'Responsibilities',\n",
       " 'Talent Management Demo Tools',\n",
       " 'Registration Dashboard',\n",
       " 'Afficher mon calendrier',\n",
       " \"Manage Associates' Pay Data\",\n",
       " 'Update Goals / Objectives',\n",
       " 'Configure Career Profile',\n",
       " 'Extended Benefits Enrolment',\n",
       " 'Afficher les anniversaires de travail',\n",
       " \"Gérer les préférences de notification d'anniversaire/de consentement\",\n",
       " \"Afficher le tableau d'emplois\",\n",
       " 'lmage Library',\n",
       " 'Afficher et gérer les avantages sociaux du collaborateur',\n",
       " 'View Your Pay',\n",
       " 'Manage Advanced Workflows',\n",
       " 'Modifier mes données démographiques personnelles',\n",
       " 'Manage Payroll Policies',\n",
       " 'Configure Competencies',\n",
       " 'Create/Update/Delete Associate Data',\n",
       " 'View Team Check-Ins',\n",
       " 'Configure Leave of Absence',\n",
       " 'travel blog',\n",
       " 'Complete Engagement Pulse',\n",
       " 'Importer les données de configuration du système',\n",
       " 'Create a Survey',\n",
       " 'Organisation Management',\n",
       " 'Afficher vos tâches d’accueil et intégration',\n",
       " 'Gérer vos successeurs',\n",
       " 'View Classic Recruiting Management',\n",
       " 'Azure AD Dashboard',\n",
       " \"Cessation d'emploi en masse\",\n",
       " 'Panel de control de registro',\n",
       " 'Manage Compensation Reviews',\n",
       " 'Licenses & Certifications Dashboard',\n",
       " 'Configure Unions',\n",
       " 'Gérer les signalements de maladie et de blessure',\n",
       " 'Outbound Integrations Execution Dashboard',\n",
       " 'Manage Broadcast(s) - Voice of the Employee',\n",
       " \"Voir l'emplacement des bureaux\",\n",
       " 'Performance Settings',\n",
       " 'Agregar mi dirección postal',\n",
       " 'Configure Approval Request Workflow',\n",
       " 'Licences & Certifications Dashboard',\n",
       " 'Go to BVCore',\n",
       " 'My Courses',\n",
       " 'Configure Compensation Codes',\n",
       " 'actionlinkLFtest',\n",
       " 'Afficher les rapports de congés',\n",
       " 'View Your Onboarding',\n",
       " 'Afficher Wisely',\n",
       " 'Importer les données du travailleur',\n",
       " 'Policy Management',\n",
       " 'Outils de base pour l’intégration',\n",
       " 'Process Internal Hires',\n",
       " 'Ajouter mon adresse physique',\n",
       " 'Enter Hours Worked',\n",
       " 'Ver y gestionar beneficios del colaborador',\n",
       " 'Responsabilidades',\n",
       " 'Bulk Termination',\n",
       " 'Termination - Configuration Settings',\n",
       " 'Create New Client',\n",
       " 'View Teams',\n",
       " 'Company Links and FAQs',\n",
       " 'Mettre à jour mes numéros de téléphone',\n",
       " 'Core HR Inbound Import Staging List',\n",
       " \"Voir les unités d'entreprise\",\n",
       " 'Consulter votre paie',\n",
       " 'Create Criteria',\n",
       " 'Manage Settings',\n",
       " 'Tableau de bord de la surveillance IDX Azure',\n",
       " 'Learning Management',\n",
       " 'Bulk Unit Management',\n",
       " 'Work Authorization Type Configuration',\n",
       " 'Manage Licenses and Certifications Library',\n",
       " 'Bulk Job Change',\n",
       " 'Gérer les autorisations',\n",
       " 'Manage Journeys Deprecated',\n",
       " 'Agregar mis contactos personales',\n",
       " 'View My Associate Profile',\n",
       " 'View Client Management',\n",
       " 'View Legal Entities',\n",
       " 'My Learning',\n",
       " 'Modifier les détails de la date de naissance',\n",
       " 'Registros de empleo',\n",
       " \"View Team's Snapshot\",\n",
       " 'Access ADPRM Recruiting Dashboard',\n",
       " 'Global Integration Dashboard',\n",
       " 'View My Pay Statements',\n",
       " 'Tableau de bord de gestion des approbations',\n",
       " 'View Team Check-ins',\n",
       " \"View and Manage Associates' Benefits\",\n",
       " 'Work Authorisation Type Configuration',\n",
       " 'Manage your Successors',\n",
       " 'Configure Talent Cycles',\n",
       " 'View Criteria Control Center',\n",
       " 'View My Org',\n",
       " 'View Fluid Field Activity Logs',\n",
       " 'Manage Timecards',\n",
       " 'Manage Journeys',\n",
       " 'View and Manage Benefits',\n",
       " 'Reports & Analytics',\n",
       " 'Compliance on Demand',\n",
       " 'Manage EV5 Integration',\n",
       " 'Manage Illness and Injury Reporting',\n",
       " 'View Your Teams',\n",
       " 'Integrations Dashboard',\n",
       " 'Performance Evaluation',\n",
       " 'Generate Cornerstone Files Manually',\n",
       " \"View Your Company's News\",\n",
       " 'View support tickets',\n",
       " 'Administrar el tiempo libre de la empresa',\n",
       " 'Lifion Generative AI',\n",
       " 'Afficher les horaires de l’équipe',\n",
       " 'Export Associate Details',\n",
       " 'Configure Movement Reasons',\n",
       " 'Afficher ma fiche de présence',\n",
       " 'View Additional Time Features',\n",
       " \"Tableau de bord Datacloud de l'administrateur\",\n",
       " 'Change Login Methods',\n",
       " 'Manage Internal Mobility Pulse',\n",
       " 'Manage Team Onboarding',\n",
       " 'Manage Company Policies',\n",
       " 'View/Edit or Approve Team Time Card',\n",
       " 'Ver registros de actividad de valores de campo fluido',\n",
       " 'Configure Follow-up Actions',\n",
       " 'View My Identifications',\n",
       " 'View Pulses To Complete',\n",
       " 'Organization Management',\n",
       " 'Manage Time',\n",
       " 'View Wisely-Flow',\n",
       " 'Ver movimientos de las personas',\n",
       " 'View Recruiting Overview',\n",
       " 'Complete Check-In',\n",
       " 'Terminer la mise au point',\n",
       " 'Clock In/Out',\n",
       " 'View My Direct Deposits',\n",
       " 'Configurer les congés autorisés',\n",
       " 'View Client Provided Translations',\n",
       " 'Pay Statement via Company Links',\n",
       " 'Agregar mi dirección física',\n",
       " 'Congés en vrac',\n",
       " 'View My Snapshot',\n",
       " 'Manage Payroll Calendar',\n",
       " 'Afficher le tableau de bord Paie',\n",
       " 'Associate data import',\n",
       " 'Configurer les buts/objectifs',\n",
       " 'Pick up Shift',\n",
       " 'Ajouter mes coordonnées personnelles',\n",
       " 'Manage New Hires from ATS',\n",
       " \"Pointage à l'arrivée / au départ\",\n",
       " 'Associate Directory',\n",
       " 'Gérer le congé',\n",
       " 'Afficher mon Org',\n",
       " 'IDX Azure Monitoring Dashboard',\n",
       " 'View Leave Requests & Balances',\n",
       " 'Importer les données du collaborateur',\n",
       " 'Gérer le tableau de bord eI-9',\n",
       " 'Modifier mes nationalités',\n",
       " 'View My Check-Ins',\n",
       " 'Calibrate Talent Attributes',\n",
       " 'Étalonner les attributs du talent',\n",
       " 'Manage Recruiting',\n",
       " 'Mettre à jour mon adresse de courriel',\n",
       " 'Consulter et demander un congé autorisé',\n",
       " 'Import Client Provided Translations',\n",
       " 'Recognize Someone',\n",
       " 'View Completed Engagement Pulses',\n",
       " 'Manage global payroll',\n",
       " 'Tableau de bord DataCloud',\n",
       " 'Ver su pago',\n",
       " 'Configure Worker Compensation Codes',\n",
       " 'Configure Work Relationship Types',\n",
       " 'Garnishments Dashboard',\n",
       " 'Voir les nouvelles de l’entreprise',\n",
       " 'Image Library',\n",
       " 'Configurer la classification des associés',\n",
       " \"Afficher les familles d'emploi\",\n",
       " 'Award Management',\n",
       " 'ESS Training',\n",
       " 'View and Request Leave of Absence',\n",
       " 'View and Approve Team Timecards',\n",
       " 'View Payroll Dashboard',\n",
       " 'Manage eI-9 Dashboard',\n",
       " \"Gérer les congés de l'entreprise\",\n",
       " 'Configuration du type de permis de travail',\n",
       " 'View My Licences & Certifications',\n",
       " 'Manage/Update Talent Assessments',\n",
       " \"Modifier l'accès aux produits\",\n",
       " 'Configure Recruiting',\n",
       " 'Configure Unions and Bargaining Units',\n",
       " 'Reporting Help',\n",
       " 'View Work Anniversaries',\n",
       " 'Ajouter mon adresse postale',\n",
       " 'View People Movements',\n",
       " 'Afficher vos avantages sociaux',\n",
       " 'Complete My Return To Workplace Survey',\n",
       " 'View My Licenses & Certifications',\n",
       " 'Gérer le tableau de bord du rapport ACA',\n",
       " 'Manage Pay Group Transfer',\n",
       " 'Afficher et approuver les fiches de présence de l’équipe',\n",
       " 'View organization information for Salvador Dali',\n",
       " 'Tableau de bord des accommodements spéciaux',\n",
       " 'Asset Management',\n",
       " 'View and Approve Team Time Off',\n",
       " 'Configure Onboarding',\n",
       " 'Import Associate Data',\n",
       " 'Gestionar el consentimiento para los puntos de referencia de la ADP',\n",
       " 'Performance Configuration',\n",
       " 'Modifier l’image du profil',\n",
       " 'Manage Pay Runs',\n",
       " 'Create Position',\n",
       " 'DHModalsize',\n",
       " 'Lightweight Integrations',\n",
       " 'Manage Record of Employment',\n",
       " 'View Cost Centres',\n",
       " 'View Engagement Pulse Results',\n",
       " 'vamsi demo link',\n",
       " 'Afficher le tableau de bord de synchronisation des employés de Cornerstone',\n",
       " 'Track Applications',\n",
       " 'Add My Mailing Address',\n",
       " 'Manage Payroll Company Data',\n",
       " 'View My Tax Withholding',\n",
       " 'Tablero de retenciones',\n",
       " 'Afficher votre rémunération',\n",
       " 'Manage My Career Development',\n",
       " 'Afficher et approuver les congés de l’équipe',\n",
       " 'Ver puestos',\n",
       " 'Importer les valeurs des champs fluides',\n",
       " 'Solicitar tiempo libre',\n",
       " 'Manage My Licenses or Certifications',\n",
       " 'Document Management Dashboard',\n",
       " 'Benefits Implementation',\n",
       " 'View My Timecard',\n",
       " 'Records of Employment',\n",
       " 'Afficher le tableau de bord des entrevues de départ',\n",
       " \"View Team's SnapShot\",\n",
       " 'Gestion des données des avantages sociaux',\n",
       " 'View FAQs',\n",
       " 'Manage Leave',\n",
       " 'View Job Families',\n",
       " 'Payroll Imports',\n",
       " 'Manage Hours and Earnings',\n",
       " 'Add My Postal Address',\n",
       " 'Statement Of Remittance',\n",
       " 'View Office Locations',\n",
       " 'Configurar reconocimientos',\n",
       " 'Manage Licences and Certifications Library',\n",
       " 'Search For Jobs',\n",
       " 'Configure Associate Classification',\n",
       " 'Enviar correos electrónicos de registro',\n",
       " 'Manage EEO Reports',\n",
       " 'Ver mis reconocimientos',\n",
       " 'Change Product Access',\n",
       " 'Review My Skills',\n",
       " 'Integration Core Tooling',\n",
       " 'View Positions',\n",
       " 'Manage Authorizations',\n",
       " 'Manage Cycle Setup',\n",
       " 'Export Current Data',\n",
       " 'My Delegates Information',\n",
       " 'View Client Administrators',\n",
       " 'Tableau de bord de gestion du document',\n",
       " 'Import Pay Data',\n",
       " 'Manage My OKRs',\n",
       " 'Edit My Personal Demographics',\n",
       " 'View Exit Interviews Dashboard',\n",
       " 'Mise à jour des buts / objectifs',\n",
       " 'Inscription aux avantages sociaux complémentaires',\n",
       " 'Quarter and Year End Checklist',\n",
       " 'View StandOut Suite',\n",
       " 'All Delegates',\n",
       " 'Submit a help ticket',\n",
       " 'Voir les classifications des professions',\n",
       " 'Identify Critical Positions',\n",
       " 'AOID Tracker',\n",
       " 'Afficher le centre de contrôle des critères',\n",
       " 'Manage Company Time Off',\n",
       " 'Afficher vos équipes',\n",
       " 'Configurar los recordatorios de caducidad de permiso de trabajo',\n",
       " 'User Management Troubleshooting',\n",
       " 'Configure Journeys',\n",
       " 'Manage Survey(s) - Voice of the Employee',\n",
       " 'Tableau de bord eI-9',\n",
       " 'View Cost Centers',\n",
       " 'View My Check-ins',\n",
       " 'Special Accommodation Dashboard',\n",
       " 'Send Registration Emails',\n",
       " 'View Criteria Control Centre',\n",
       " 'Document Cloud',\n",
       " 'Manage ACA Report Dashboard',\n",
       " \"Tableau de bord de l'exécution des intégrations sortantes\",\n",
       " \"View My Reports' OKRs\",\n",
       " 'View Dynamic Teams Chart',\n",
       " 'Edit My Nationalities',\n",
       " \"Manage Your Company's Compensation\",\n",
       " 'Recruiting Configuration',\n",
       " \"Gérer les formulaires d'avantages sociaux et les documents de régime\",\n",
       " 'Ajouter une nouvelle embauche',\n",
       " 'tabaccesstestcase',\n",
       " 'View Engagement Pulse',\n",
       " 'Ver las noticias de la compañía',\n",
       " 'Import Fluid Field Values',\n",
       " 'Bulk Import Custom Fields',\n",
       " 'Configure Recognitions',\n",
       " 'View HR Inbound Data Dashboard',\n",
       " 'Manage Succession Plans',\n",
       " 'Crear puesto',\n",
       " 'Administración de datos de beneficios',\n",
       " 'Import Worker Data',\n",
       " 'Rastreador de AOID',\n",
       " 'Retrospective',\n",
       " 'Sync Coded Fields Table',\n",
       " 'My Team Learning',\n",
       " 'View Team Chart',\n",
       " 'Solución de problemas de gestión de usuarios',\n",
       " 'Afficher les entités juridiques',\n",
       " 'Afficher mon profil de collaborateur',\n",
       " 'Afficher tableau de bord données entrantes RH',\n",
       " 'Create News Post',\n",
       " 'Declaración de remesa',\n",
       " 'Configure Work Authorization Expiration Reminders',\n",
       " \"Afficher l'évaluation de votre rendement\",\n",
       " 'Configure Career Development',\n",
       " 'Broadcast Message(s) - Voice of the Employee',\n",
       " 'Submit a Performance Pulse',\n",
       " 'Configure Advanced Workflows',\n",
       " 'Integration Dashboard',\n",
       " 'Take my Strengths Assessment',\n",
       " 'Demander un congé',\n",
       " 'Configure Performance Plans',\n",
       " 'View Your Compensation',\n",
       " 'Tableau de bord de migration des documents en masse',\n",
       " 'Afficher mes congés',\n",
       " 'View Your Surveys',\n",
       " 'Create Requisition Request',\n",
       " 'View Company Org Chart',\n",
       " \"Tableau de bord d'embauche\",\n",
       " 'View Enterprise Units',\n",
       " 'Import New Hires',\n",
       " 'View Time Reports',\n",
       " 'Gestionar usuarios',\n",
       " 'Manage VETS 4212 Reports',\n",
       " 'View Your Dynamic Teams',\n",
       " 'Update My Email',\n",
       " 'Créer des critères',\n",
       " 'Manage Return to Workplace Surveys',\n",
       " 'Assign Additional Associate Identifiers',\n",
       " 'Add New Hire',\n",
       " 'Add My Physical Address',\n",
       " 'Manage News Posts',\n",
       " 'View My Time Off',\n",
       " 'Google',\n",
       " 'Manage Benefits Supplemental Fields',\n",
       " 'Launch Engagement Pulse',\n",
       " 'Manage Consent for ADP Benchmarks',\n",
       " 'Gérer les exécutions de paie',\n",
       " 'Tableau de bord des enregistrements',\n",
       " 'View organization information for Pro Check One',\n",
       " 'Apex Inbound Integration',\n",
       " 'Configure Goals / Objectives',\n",
       " 'Bulk Document Migration Dashboard',\n",
       " 'View Internal Career Site',\n",
       " 'Edit Birth Details',\n",
       " 'Afficher les postes',\n",
       " 'Edit Profile Image',\n",
       " 'Afficher mes reconnaissances',\n",
       " 'Survey(s) - Voice of the Employee',\n",
       " 'Reconocer a alguien',\n",
       " 'Panel de control eI-9',\n",
       " 'Aller à BVCore',\n",
       " 'View Team Schedules',\n",
       " 'Manage Users',\n",
       " \"Gérer les données de l'entreprise de paie\",\n",
       " 'View My Check-In',\n",
       " 'disabled TALACQ-55299',\n",
       " 'Manage Successors for Positions',\n",
       " 'Afficher et gérer les avantages sociaux',\n",
       " 'Afficher mes enregistrements',\n",
       " 'View Company News',\n",
       " 'Integrations Health Monitor',\n",
       " 'DataCloud Dashboard',\n",
       " \"View Everyone's OKRs\",\n",
       " 'View HR Dashboard',\n",
       " 'Gestion des actifs',\n",
       " 'Manage Payload Configuration',\n",
       " 'StandOut Champion Central',\n",
       " 'Acceso a cambio de producto',\n",
       " 'Informes y análisis',\n",
       " 'Afficher les fonctions supplémentaires de Time',\n",
       " 'Ver el panel de control de sincronización de empleado de Cornerstone',\n",
       " 'Extended Benefits Enrollment',\n",
       " 'Manage Talent Cycles',\n",
       " 'Performance Setup',\n",
       " 'Manage Cornerstone Configuration',\n",
       " 'Panel de control de licencias y certificaciones',\n",
       " 'Afficher les équipes',\n",
       " 'Import Legacy System Data',\n",
       " 'Classroom Training Management',\n",
       " 'View Your Performance Reviews',\n",
       " 'View My SnapShot',\n",
       " 'Configure Work Authorisation Expiration Reminders',\n",
       " 'Import New Hires from ATS',\n",
       " 'Manage Birthday Notification/Consent Preferences',\n",
       " 'Complete Check-in',\n",
       " 'Afficher les anniversaires',\n",
       " 'Benefits Data Management',\n",
       " 'Import System Configuration Data',\n",
       " 'Gérer les politiques d’entreprise',\n",
       " 'Manage Disciplinary Actions',\n",
       " 'Approval Management Dashboard',\n",
       " 'Execute Integration',\n",
       " \"Afficher les modèles d'emploi\",\n",
       " 'Case Management',\n",
       " 'Bulk Leave of Absence',\n",
       " 'Configure Additional Associate Identifiers',\n",
       " 'View My Recognitions',\n",
       " 'Manage General Ledger',\n",
       " 'Afficher un instantané remarquable pour Qi Cong Zhang',\n",
       " 'View My Schedule',\n",
       " 'Administrar datos de pago de colaboradores',\n",
       " 'Gérer le calendrier de paie',\n",
       " \"Voir les résultats de la mesure de l'engagement\",\n",
       " 'Prior Quarter Adjustment',\n",
       " 'Hiring Dashboard',\n",
       " 'Manage Authorisations',\n",
       " 'Ver Mis retenciones fiscales',\n",
       " 'Responsabilités',\n",
       " 'View Dynamic Teams',\n",
       " 'View Cornerstone Employee Sync Dashboard',\n",
       " 'Import Education History',\n",
       " 'Configure Compensation Reviews',\n",
       " 'Request Time Off',\n",
       " 'Audit Reporting',\n",
       " 'Add My Personal Contacts',\n",
       " 'Yahoo',\n",
       " 'View Occupational Classifications',\n",
       " 'Edit My Identification',\n",
       " 'Enviar un pulso de rendimiento',\n",
       " 'Access Internal Career/Job Site',\n",
       " 'One LMS Test',\n",
       " 'Gérer la rémunération de votre entreprise',\n",
       " 'View And Manage Fluid Fields',\n",
       " 'Create Cycle',\n",
       " 'Recognise Someone',\n",
       " 'Manage Benefits forms and plan documents',\n",
       " 'Celergo Dashboard',\n",
       " 'Manage Fluid Field Definitions',\n",
       " 'Access Absorb LMS',\n",
       " 'Manage My Licences or Certifications',\n",
       " 'Update My Phone Numbers',\n",
       " 'Manage Central Rate',\n",
       " 'Datacloud Admin Dashboard',\n",
       " 'Ver mis licencias y certificaciones']"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Initialize the evaluator with corrected class\n",
    "evaluator = EmbeddingSimilarityEvaluator(\n",
    "    sentences1=,\n",
    "    sentences2=,\n",
    "    scores=validation_dataset[\"score\"],\n",
    "    main_similarity='COSINE',\n",
    "    batch_size=16,\n",
    "    name=\"similarity-eval\",\n",
    "    show_progress_bar=True,\n",
    "    write_csv=True,\n",
    "    precision=\"float32\"\n",
    ")\n",
    "\n",
    "# Run the evaluation\n",
    "metrics = evaluator(output_path=\".\", epoch=1, steps=100)\n",
    "\n",
    "# Display the results\n",
    "print(\"Evaluation Metrics:\", metrics)"
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
