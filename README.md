NeuroDOT_py README


1. Installation:

	1. First, download the latest version of Python from: https://www.python.org/downloads/
	
	2. Download VSCode: https://code.visualstudio.com
	
	3. Download the Jupyter notebook extension for VSCode: launch your VS Code and type “jupyter notebook” in the extension search box. Select the first result (Jupyter) and click 'Install'.
	
	4. Download the code from the NeuroDOT_py GitHub repository: Navigate to https://github.com/WUSTL-ORL/NeuroDOT_py, and click the green box on the right side of the page that says "Code." From the choices, select 'Download ZIP' and your download will begin. Once the .zip file is downloaded, extract the folder 'NeuroDOT_py-main' to your working directory. 


2. Geting Started
		
	1. The toolbox contains 4 folders: Data, neuro_dot, Support Files, and outputfiles/output_Images.
	
		1. The Data folder contains 10 data samples including both retinotopic mapping of visual cortex and mapping of hierarchical language processing with HD-DOT. There are also two example parameter files, 'params.txt,' and 'params2.txt' to be used with 'getting_started' (the NeuroDOT Preprocessing script).
             
		2. The neuro_dot folder contains the library, consisting of modules for each category of function involved in NeuroDOT_py (Analysis, File_IO, Light Modeling, Matlab   Equivalent Functions, Reconstruction, Spatial Transforms, Temporal Transforms, and Visualizations). There is also a function named DynamicFilter, which is used in 'getting_started.ipynb' to simplify visualizations for data pre-processing. There is also 'requirements.txt' which contains all of the necessary libraries to be installed to use NeuroDOT_py.	
		
		3. The Support Files folder contains necessary files for running NeuroDOT pipelines.
			- The A matrix file required for Reconstruction is too large to be posted on GitHub, so it can be downloaded from: https://www.nitrc.org/projects/neurodot/. Other A matrices will be added in the future.
	     
		4. The 'outputfiles' folder is created after running 'getting_started' and is where all of the images (.png) generated will be saved to.
	     
	2. 'getting_started.ipynb' in the main folder is the Jupyter notebook for running the NeuroDOT Pre Processing Script. This is the file that you will open in VSCode/Jupter Notebook to run and manipulate the code. 
 

