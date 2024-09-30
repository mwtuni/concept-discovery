This project employs Generative AI to create new concepts about anything. It generates thousands of images based on user-defined idea and reverse-engineers the technologies, materials, or other specifics visible in these generated concept images.

![image](https://github.com/user-attachments/assets/bd5d5fd9-f5a4-4bfb-b42b-948732d0545c)

### Architectural Overview of GenAI-Assisted Concept Discovery Tool ###

1. **Stable Diffusion (SD)**: 
   - Uses "Image Generative AI (AUTO1111)" to generate thousands of images based on user-defined specifications for new products and serves as the primary creative engine in this workflow.

2. **Vision-Language Model (VLM)**:
   - The VLM, LLaVA (Vision-Language Model), is enhanced with Retrieval-Augmented Generation (RAG) information. This additional information enables LLaVA to recognize and search for technologies that were not originally part of its training, expanding its capability to analyze novel features in the generated images.

3. **Large Language Model (LLM)**:
   - The LLM, codellama, receives the same RAG information as the VLM. It uses this information to list and detail the specific technologies identified by the VLM, exporting them into a structured CSV format for subsequent data analysis.

4. **Data Analysis & Interactive Scatter Plot View**:
   - The CSV data provided by the LLM is analyzed using Python libraries (Pandas, Seaborn, Numpy). The main outcome is an **interactive Scatter Plot view** that allows users to explore and observe technological details. When a user hovers over a particular data point in the plot, detailed information about the technologies present is displayed, providing an intuitive way to analyze and understand the concepts discovered through this Generative AI process.
![image](https://github.com/user-attachments/assets/c4c8001f-4401-43e7-aabb-dae5a8b4d79e)
 
This architecture leverages a combination of image generation, vision-language understanding, and data analysis to facilitate a comprehensive concept discovery process, enabling users to visualize and analyze technologies embedded in generated product images.

## Multimodal data integrity verification

The **jenkins** folder in the GitHub repository contains a multimodal sanity check designed to validate the integration of vision, language, and Retrieval-Augmented Generation (RAG) capabilities in the algorithm 

### Verification Process

The sanity check involves prompting the model with the question, "What is the character in the image and what hygienic product does he use?" The correct answer should integrate:
- Visual analysis from the image,
- Information retrieved from the `data.txt` file,
- Details from the user prompt.

If the system is functioning correctly, it should provide an accurate response that includes elements from all three sources. If there is an issue with the model's ability to utilize these sources, it might produce hallucinated or incomplete answers, indicating a breakdown in multimodal integration.

### Purpose

The Jenkins sanity check is crucial for validating that the LLaVA VLM effectively uses both visual and RAG data in conjunction with user prompts, ensuring that the system responds accurately and without unnecessary hallucination. This test helps maintain the integrity and robustness of the AI-assisted concept discovery tool by confirming that it can handle complex, multimodal data inputs.
