# PaddleOCR Pretraining and Inference Code

This repository is structured to facilitate two main tasks: **data generation for PaddleOCR pretraining** and **inference using trained models**. Below is a detailed description of the repository contents, along with sequential guidelines to assist in the effective use of the code.

## Repository Structure

### Data Generation for Pretraining

- **`fonts/`**: Contains various text fonts used to introduce text variations during synthetic data generation.
- **`jsons/`**: Stores JSON files containing text data of PNIDs extracted via GoogleOCR.
- **`pnid_images/`**: Includes images converted from PNIDs.

### Scripts and Utilities

- **`append_data.py`**: A utility to iteratively append images and their corresponding text annotations into the `mapping.txt` file for synthetic data generation.
- **`detection_data_gen.py`**: Generates data in the required format for training PaddleOCR on text detection tasks.
- **`recog_bbox_text.py`**: Creates data formatted for training PaddleOCR on text recognition tasks using PNID images and associated JSONs.
- **`synthetic_gen.py`**: A utility to generate synthetic data with text variations for recognition tasks.
- **`combined_data_gen.py`**: An end-to-end script for generating data suitable for both recognition and detection pretraining.

### Pretrained Models

- **`cls/`, `detection_model/`, `recognition_model/`**: Directories containing the latest pretrained models for classification, detection, and recognition, respectively.

### Inference and Post-Processing

- **`en_dict.txt`**: A list of symbols, numbers, alphabets, and other characters used during pretraining.
- **`clean.py`**: A post-processing utility that cleans up predictions by merging overlapping bounding boxes and removing extra boxes.
- **`combine_logic.py`**: A newly developed, easy-to-understand algorithm to combine predicted text. The accuracy of this algorithm ranges from 96% to 100%, depending on the selected threshold value.
- **`full_inference.py`**: A complete end-to-end inference script that utilizes both the newly trained recognition and detection models. The output is saved into a DataFrame for further analysis.

## Sequential Guidelines

1. **Data Generation**: Use the `combined_data_gen.py` script to generate data for both text recognition and detection tasks.
   
2. **Transfer Data**: After generating the data, transfer the `.txt` files along with the detection patches and recognition text images to their respective EC2 instances for detection and recognition tasks.

3. **Model Training**: To train the models, refer to the commands documented in the `Plan Notes.txt` file. Alternatively, you can follow the guidelines provided in the [PaddleOCR documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/finetune_en.md) for finetuning.

4. **Export Inference Model**: Once training is complete, export the trained model for inference purposes.

5. **Run Inference and Post-Processing**: Execute the following scripts in sequence to perform inference and clean up the predictions:
    - `full_inference.py`
    - `clean.py`
    - `combine_logic.py`
    - `clean.py` (run this script again for final clean-up)
