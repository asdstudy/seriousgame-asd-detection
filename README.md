# Towards Screening of Children Students with Autism Spectrum Disorder Based on Executive Functions with Serious Game and Machine Learning Approaches

This repository provides the implementation and dataset related to a study focused on the early screening of children with Autism Spectrum Disorder (ASD), using features based on executive functions extracted from serious game performance. Multiple machine learning models were evaluated to support ASD tracking strategies in educational contexts.

## Contents

- `src/`: Source code for data processing, model training, and evaluation.
- `data/`: Anonymized dataset used in the study.
- `figures/`: Graphs and curves resulting from the experiment.
- `README.md`: Project overview and usage instructions.

## Data description

To compose the database, data collection was carried out over three months in three schools located in Recife, Brazil, involving 12 children—six of whom were diagnosed with ASD, while the other six had no diagnosis. During this period, data from 640 game sessions were collected and preprocessed, distributed across three levels of difficulty, as shown in the following table, which presents the distribution of sessions by difficulty level and group (with and without ASD), and in the section "Description of data by children," which provides a detailed breakdown of the number of sessions per child.

| **Game**           | **Level** | **With ASD** | **Without ASD** |
|--------------------|-----------|--------------|-----------------|
| Memory Game        | Easy      | 78           | 73              |
|                    | Medium    | 35           | 39              |
|                    | Hard      | 23           | 35              |
| Shadow Matching    | Easy      | 44           | 50              |
|                    | Medium    | 61           | 52              |
|                    | Hard      | 19           | 27              |
| Painting Game      | Easy      | 34           | 29              |
|                    | Medium    | 22           | 14              |
|                    | Hard      | 4            | 11              |

Through the previously described interactions, 16 features were collected to compose the dataset. These observations focused on identifying behaviors that could indicate executive dysfunctions in children during the game sessions, facilitating the differentiation between individuals with and without ASD. The observed features are presented in the table.

| **Field Name**           | **Description**                                                        | **Data Type** | **Example**                          |
|--------------------------|------------------------------------------------------------------------|---------------|---------------------------------------|
| Seconds                  | Duration of the session in seconds                                     | Integer       | 205                                   |
| Game                     | The game played in the session                                         | Text          | Memory                                |
| Difficulty               | Difficulty level of the game                                           | Text          | Hard                                  |
| Child                    | Child identifier                                                       | UUID          | 2ee1f826-4520-4fc7-ae57-9349047a7328 |
| Gender                   | Child's gender                                                         | Text          | Male                                  |
| Class                    | Child's class/group                                                    | Text          | G3                                    |
| Moves                    | Total number of moves in the session                                   | Integer       | 52                                    |
| Correct                  | Total number of correct answers in the session                         | Integer       | 12                                    |
| Errors                   | Total number of errors in the session                                  | Integer       | 24                                    |
| Helps                    | Total number of help requests in the session                           | Integer       | 7                                     |
| Music_on                 | Whether the child enabled music during the session                     | Boolean       | 0                                     |
| Music_off                | Whether the child disabled music during the session                    | Boolean       | 0                                     |
| Consecutive_correct      | Highest number of consecutive correct answers in the session           | Integer       | 2                                     |
| Consecutive_errors       | Highest number of consecutive errors in the session                    | Integer       | 9                                     |
| Consecutive_helps        | Highest number of consecutive help requests in the session             | Integer       | 7                                     |
| ASD                      | Indicator if the child has ASD (1 yes, 0 no)                           | Boolean       | 1                                     |

It is important to highlight that the children were gradually introduced to the activities during the data collection process. This approach aimed to provide a more sensitive experience to track the progression or regression of their skills. The inclusion of children without a diagnosis was intended to build a balanced dataset, enabling effective monitoring of functional development.
Finally, the dataset was organized into three subsets to ensure it could be applied to machine learning models. The validation set represented 20% of the total sample, while the remaining 80% was split into 70% for training and 30% for testing.

## Description of data by children

### Children with ASD

**Child `2ee1f826-4520-4fc7-ae57-9349047a7328`**
| **Game**              | **Level** | **Sessions** |
|-----------------------|-----------|--------------|
| Memory                | Easy      | 14           |
| Memory                | Hard      | 7            |
| Memory                | Medium    | 5            |
| Painting              | Easy      | 11           |
| Painting              | Hard      | 3            |
| Pairing with shadows  | Easy      | 8            |
| Pairing with shadows  | Hard      | 3            |
| Pairing with shadows  | Medium    | 7            |

**Child `7a59cbed-5b30-4f8e-a365-93b9071dd4b8`**
| **Game**              | **Level** | **Sessions** |
|-----------------------|-----------|--------------|
| Memory                | Easy      | 19           |
| Memory                | Hard      | 4            |
| Memory                | Medium    | 4            |
| Painting              | Easy      | 4            |
| Painting              | Medium    | 2            |
| Pairing with shadows  | Easy      | 8            |
| Pairing with shadows  | Hard      | 2            |
| Pairing with shadows  | Medium    | 8            |

**Child `98f62a65-4dbf-4377-96ed-9bb90050a6f2`**
| **Game**              | **Level** | **Sessions** |
|-----------------------|-----------|--------------|
| Memory                | Easy      | 13           |
| Memory                | Medium    | 6            |
| Painting              | Easy      | 4            |
| Painting              | Medium    | 16           |
| Pairing with shadows  | Easy      | 7            |
| Pairing with shadows  | Hard      | 4            |
| Pairing with shadows  | Medium    | 6            |

**Child `b54b5fc4-4c3a-4c12-9718-b146a223e59a`**
| **Game**              | **Level** | **Sessions** |
|-----------------------|-----------|--------------|
| Memory                | Easy      | 14           |
| Memory                | Hard      | 5            |
| Memory                | Medium    | 8            |
| Painting              | Easy      | 5            |
| Painting              | Hard      | 1            |
| Painting              | Medium    | 4            |
| Pairing with shadows  | Easy      | 12           |
| Pairing with shadows  | Hard      | 2            |
| Pairing with shadows  | Medium    | 14           |

**Child `d351a361-fc42-4d85-94a6-9a571d7ea825`**
| **Game**              | **Level** | **Sessions** |
|-----------------------|-----------|--------------|
| Memory                | Easy      | 11           |
| Memory                | Medium    | 4            |
| Painting              | Easy      | 4            |
| Pairing with shadows  | Easy      | 5            |
| Pairing with shadows  | Hard      | 3            |
| Pairing with shadows  | Medium    | 20           |

**Child `f0ea0108-c3e1-471a-92dc-4eeb8be5bf0b`**
| **Game**              | **Level** | **Sessions** |
|-----------------------|-----------|--------------|
| Memory                | Easy      | 7            |
| Memory                | Hard      | 7            |
| Memory                | Medium    | 8            |
| Painting              | Easy      | 6            |
| Pairing with shadows  | Easy      | 4            |
| Pairing with shadows  | Hard      | 5            |
| Pairing with shadows  | Medium    | 6            |

### Children without ASD

**Child `427a121d-cb40-4521-9c73-49467e10e2a0`**
| **Game**              | **Level** | **Sessions** |
|-----------------------|-----------|--------------|
| Memory                | Easy      | 12           |
| Memory                | Medium    | 1            |
| Painting              | Easy      | 4            |
| Pairing with shadows  | Easy      | 9            |
| Pairing with shadows  | Hard      | 7            |
| Pairing with shadows  | Medium    | 6            |

**Child `676922bf-d986-43ec-8d42-c9eafc005a19`**
| **Game**              | **Level** | **Sessions** |
|-----------------------|-----------|--------------|
| Memory                | Easy      | 11           |
| Memory                | Hard      | 5            |
| Memory                | Medium    | 5            |
| Painting              | Easy      | 8            |
| Painting              | Hard      | 1            |
| Painting              | Medium    | 1            |
| Pairing with shadows  | Easy      | 9            |
| Pairing with shadows  | Hard      | 2            |
| Pairing with shadows  | Medium    | 6            |

**Child `72e9773e-fb52-45dd-be02-8dd133604862`**
| **Game**              | **Level** | **Sessions** |
|-----------------------|-----------|--------------|
| Memory                | Easy      | 19           |
| Memory                | Hard      | 6            |
| Memory                | Medium    | 11           |
| Painting              | Easy      | 7            |
| Painting              | Hard      | 2            |
| Painting              | Medium    | 6            |
| Pairing with shadows  | Easy      | 9            |
| Pairing with shadows  | Hard      | 6            |
| Pairing with shadows  | Medium    | 7            |

**Child `b0e37b90-8849-4c84-afce-56f2de2c10da`**
| **Game**              | **Level** | **Sessions** |
|-----------------------|-----------|--------------|
| Memory                | Easy      | 4            |
| Memory                | Hard      | 3            |
| Memory                | Medium    | 6            |
| Painting              | Easy      | 3            |
| Painting              | Hard      | 5            |
| Painting              | Medium    | 3            |
| Pairing with shadows  | Easy      | 3            |
| Pairing with shadows  | Hard      | 6            |
| Pairing with shadows  | Medium    | 8            |

**Child `b42b5ed8-8625-4328-8251-6a0d3de64207`**
| **Game**              | **Level** | **Sessions** |
|-----------------------|-----------|--------------|
| Memory                | Easy      | 12           |
| Memory                | Hard      | 6            |
| Memory                | Medium    | 9            |
| Painting              | Easy      | 5            |
| Painting              | Hard      | 3            |
| Painting              | Medium    | 2            |
| Pairing with shadows  | Easy      | 9            |
| Pairing with shadows  | Hard      | 4            |
| Pairing with shadows  | Medium    | 9            |

**Child `c568a1a9-c4d1-45f3-a0aa-0719cce0e5ed`**
| **Game**              | **Level** | **Sessions** |
|-----------------------|-----------|--------------|
| Memory                | Easy      | 15           |
| Memory                | Hard      | 5            |
| Memory                | Medium    | 7            |
| Painting              | Easy      | 2            |
| Painting              | Medium    | 2            |
| Pairing with shadows  | Easy      | 11           |
| Pairing with shadows  | Hard      | 2            |
| Pairing with shadows  | Medium    | 16           |

