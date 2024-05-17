<!-- Start Badges -->
<p align="center">
  <a href="#">
    <img src="https://badges.pufler.dev/visits/azaryasph/Improving-Employee-Retention-By-Predicting-Employee-Attrition-Using-Machine-Learning" alt="Visits Badge">
    <img src="https://badges.pufler.dev/updated/azaryasph/Improving-Employee-Retention-By-Predicting-Employee-Attrition-Using-Machine-Learning" alt="Updated Badge">
    <img src="https://badges.pufler.dev/created/azaryasph/Improving-Employee-Retention-By-Predicting-Employee-Attrition-Using-Machine-Learning" alt="Created Badge">
    <img src="https://img.shields.io/github/contributors/azaryasph/Improving-Employee-Retention-By-Predicting-Employee-Attrition-Using-Machine-Learning" alt="Contributors Badge">
    <img src="https://img.shields.io/github/last-commit/azaryasph/Improving-Employee-Retention-By-Predicting-Employee-Attrition-Using-Machine-Learning" alt="Last Commit Badge">
    <img src="https://img.shields.io/github/commit-activity/m/azaryasph/Improving-Employee-Retention-By-Predicting-Employee-Attrition-Using-Machine-Learning" alt="Commit Activity Badge">
    <img src="https://img.shields.io/github/repo-size/azaryasph/Improving-Employee-Retention-By-Predicting-Employee-Attrition-Using-Machine-Learning" alt="Repo Size Badge">
    <img src="https://img.shields.io/badge/contributions-welcome-orange.svg" alt="Contributions welcome">
    <img src="https://www.codefactor.io/repository/github/azaryasph/Improving-Employee-Retention-By-Predicting-Employee-Attrition-Using-Machine-Learning" alt="CodeFactor" />
</a>
</p>
<!-- End Badges -->

<!-- Start Project Title -->
# <img src="https://yt3.googleusercontent.com/ytc/AIdro_n0EO16H6Cu5os9ZOOP1_BsmeruHYlmlOcToYvI=s900-c-k-c0x00ffffff-no-rj" width="30"> Mini Project 5: Improving Employee Retention By Predicting Employee Attrition Using Machine Learning<img src="https://yt3.googleusercontent.com/ytc/AIdro_n0EO16H6Cu5os9ZOOP1_BsmeruHYlmlOcToYvI=s900-c-k-c0x00ffffff-no-rj" width="30">
<!-- End Project Title -->

<!-- Start Image Acc -->
![Employees](https://images.unsplash.com/photo-1504384308090-c894fdcc538d?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)
Photo by <a href="https://unsplash.com/@frantic?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Alex Kotliarskyi</a> on <a href="https://unsplash.com/photos/people-doing-office-works-QBpZGqEMsKg?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
<!-- End Image Acc -->

<!-- Start Table of Contents -->
## Table of Contents
1. [📝 Background Project](#background-project)
2. [🔭 Scope of Work](#scope-of-work)
3. [📊 Data and Assumptions](#data-and-assumptions)
4. [📈 Data Analysis](#data-analysis)
5. [🧹 Data Preprocessing](#data-preprocessing)
6. [🤖 Modelling and Evaluation](#modelling-and-evaluation)
7. [🔚 Conclusion](#conclusion)
8. [💻 Installation and Usage](#installation-and-usage)
9. [🙏 Acknowledgements](#acknowledgements)
<!-- End Table of Contents -->

<!-- Start Background Project -->
## Background Project
### What is the problem to solve?
As a member of the Data Scientist team at a technology start-up company. The company is currently experiencing big problems; many of their employees have submitted their resignations, but the company has not yer decided on this matter. I will help the company to explain the current condition of its employees, as well as explore the problems within the company that cause employees to resign, so that the company can make the right decisions to improve employee retention.


### Why is this problem important?
No matter the ecomonic climate, employee retention is critical to the success of any company. It can save the company money, foster a positive work environment, lead to better team member performance, generate more revenue, and create a more stable company culture. Employee retention is a key indicator of company health and success. It is important to understand why employees leave a company and what can be done to retain them.

### What is the goal of this project?
The goal of this project is to predict employee resignations and provide actionable business recommendations to improve employee retention. The company can use the results of this project to make the right decisions to improve employee retention.

### What Will be Changed if There are the Model Results?
If there's model predction results, the company can make the right decisions to use their resources more effectively and efficiently.
<!-- End Background Project -->

<!-- Start Scope of Work -->
## Scope of Work
### What is the scope of this project?
The scope of this project is around predicting employee resignations and providing actionable business recomendations from the data analysis and model results.

### How is the output of the developed model?
The output of the developed model is a prediction of employee resignations and actionable business recommendations to improve employee retention.
<!-- End Scope of WOrk -->

<!-- Start Data and Assumptions -->
## Data and Assumptions
### Data Size
The dataset contains 287 rows and 25 columns.

### Features and Description
|Feature|Description|
|---|---|
|Username|Employee's username|
|Enterprise ID|Employee's enterprise ID|
|StatusPernikahan|Employee's marital status|
|Jenis Kelamin|Employee's gender|
|StatusKepegawaian|Employee's employment status|
|Pekerjaan|Employee's job title|
|JenjangKarir|Employee's career level|
|PerformancePegawai|Employee's performance|
|AsalDaerah|Employee's origin|
|HiringPlatform|Employee's hiring platform|
|SkorSurveyEngagement|Employee's engagement survey score|
|SkorKepuasanPegawai|Employee's satisfaction score|
|JumlahKeikutsertaanProjek|Employee's number of project participations|
|JumlahKeterlambatanSebulanTerakhir|Employee's number of late arrivals in the last month|
|JumlahKetidakhadiran|Employee's number of absences|
|NomorHP|Employee's phone number|
|Email|Employee's email|
|TingkatPendidikan|Employee's education level|
|PernahBekerja|Employee's previous work experience|
|IkutProgramLOP|Employee's LOP program participation|
|AlasanResign|Employee's resignation reason|
|TanggalLahir|Employee's birth date|
|TanggalHiring|Employee's hiring date|
|TanggalPenilaianKaryawan|Employee's evaluation date|
|TanggalResign|Employee's resignation date|

### Assumptions
Based on my domain knowledge from the data analysis, I assume that the following features are important for predicting employee resignations:
- `PerformancePegawai`
- `SkorSurveyEngagement`
- `SkorKepuasanPegawai`
- `JumlahKeikutsertaanProjek`
- `JumlahKeterlambatanSebulanTerakhir`
- `JumlahKetidakhadiran`
- `TingkatPendidikan`
- `JenjangKarir`
- `StatusKepegawaian`
- `Pekerjaan`
- `AsalDaerah`
- `Age` (derived from `TanggalLahir`)
- `EmploymentDuration` (derived from `TanggalHiring` and `TanggalResign`, for more details please see the notebook)
- `HiringToEval` (derived from `TanggalHiring` and `TanggalPenilaianKaryawan`)
The rest of other features are not necessary for predicting employee resignations, because they are Identifiers and some of them are data leakage.
<!-- End Data and Assumptions -->

## Data Analysis


### Data Preprocessing


### Modelling and Evaluation


#### Model Selection


#### Metrics Evaluation


#### Model Evaluation


#### Model Business Impact Simulation


#### Model Business Recommendations based on Feature Importances


**Actionable Business Recommendations**



### Conclusion


### Installation and Usage
1. Clone this repository
```
git clone
```
2. Install the required libraries
```
pip install -r requirements.txt
```
3. Run the Jupyter Notebook
```
jupyter notebook
```
4. Open the Jupyter Notebook file and run the code

### Acknowledgements
<!-- Thanks to [Rakamin Academy](https://www.rakamin.com/) for providing the dataset and the opportunity to work on this project. I would also like to thank Mr. [Fiqry Revadiansyah](https://www.linkedin.com/in/fiqryrevadiansyah/) for his guidance and support throughout the project.


![Thank You GIF](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExa3BiNGlqejgxaGh0cjc3ODVzNTNtb3RhZmE5MTRyYzBvd3k2ZjQ0aCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l3q2FnW3yZRJVZH2g/giphy.gif) -->