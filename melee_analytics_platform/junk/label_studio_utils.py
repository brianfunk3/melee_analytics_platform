import os
import shutil
import zipfile

def label_studio_export(label_studio_project_num, project_name, export_format = 'voc'):
    """
    need to delete the old data, then export and move
    """
    # here's the path where things will export to, probs should be parameterized but oh well
    export_path = "C:\\Users\\brian\\AppData\\Local\\label-studio\\label-studio\\export"
    
    # first lets delete all exports with the project id
    [os.remove(export_path + "\\" + file) for file in os.listdir(export_path) if file.startswith(f'project-{label_studio_project_num}')]

    # now lets export there now
    os.system(f"label-studio export {label_studio_project_num} {export_format}")

    # get the files to move
    export_to_move = [file for file in os.listdir(export_path) if (file.startswith(f'project-{label_studio_project_num}') and file.endswith('.zip'))][0]

    # make our export location if it doesn't exist and send it
    export_location = f'projects/{project_name}/'
    if not os.path.exists(export_location):
        os.makedirs(export_location)

    # remove any existing annotations
    if os.path.exists(f'projects/{project_name}/Annotations'):
        shutil.rmtree(f'projects/{project_name}/Annotations')
    if os.path.exists(f'projects/{project_name}/images'):
        shutil.rmtree(f'projects/{project_name}/images')

    # and unzip it into our project directory
    with zipfile.ZipFile(export_path + "\\" + export_to_move, 'r') as zip_ref:
        zip_ref.extractall(export_location)