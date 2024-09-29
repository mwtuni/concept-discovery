This project employs Generative AI to create new concepts about anything. It generates thousands of images based on user-defined new product concepts and reverse-engineers the technologies, materials, or other specifics visible in these images.

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
