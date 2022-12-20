# Handling Hallucinations In Neural Semantic Parsing

Repository of the paper Handling Hallucinations In Neural Semantic Parsing.

## Structure of the Repository

The repository contains three main folders
- Hallucination Simulation Framework (HSF) 
  - Located in HSF-KQA-PRO/HallucinationSimulationFramework
  - It is used to generate the dataset which will be used by the NSP model to train and extract hallucination detection features
- NSP model
  - Located in HSF-KQA-PRO/Bart_Program
  - With that code you can train and extract
- Hallucination Detection Model
  - Located in HallucinationDetectionModel
  - This folder contains the code to train the HDM model
  - And it contains the end-to-end scorer


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


# Licensing
See [LICENSE](LICENSE) and [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES) in the root of the project.
- CC BY NC: all folders *except* HSF-KQA-PRO/HallucinationSimulationFramework/out_folder
- CC BY SA: HSF-KQA-PRO/HallucinationSimulationFramework/out_folder

# Authors
- Andrea Bacciu
- Marco Damonte
- Marco Basaldella
- Emilio Monti



