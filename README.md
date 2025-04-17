# Towards Screening of Children Students with Autism Spectrum Disorder Based on Executive Functions with Serious Game and Machine Learning Approaches

This repository provides the implementation and dataset related to a study focused on the early screening of children with Autism Spectrum Disorder (ASD), using features based on executive functions extracted from serious game performance. Multiple machine learning models were evaluated to support ASD tracking strategies in educational contexts.

## Contents

- `src/`: Source code for data processing, model training, and evaluation.
- `data/`: Anonymized dataset used in the study.
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
