import os
import subprocess


def run_COMA(path, output_path):

    all_pers = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))][8:]
    for pers in all_pers:
        pers_path = os.path.join(path, pers)
        pers_output = os.path.join(output_path, pers)
        os.makedirs(pers_output, exist_ok=True)
        meshes = [f for f in os.listdir(pers_path) if (".obj" in f) or (".ply" in f)]
        folders = [f for f in os.listdir(pers_path) if os.path.isdir(os.path.join(pers_path, f))]
        print(folders)
        for folder in folders:
            meshes += [(folder, f) for f in os.listdir(os.path.join(pers_path, folder)) if (".obj" in f) or (".ply" in f)]
        for mesh in meshes:
            if type(mesh) == tuple:
                os.makedirs(os.path.join(pers_output, mesh[0]), exist_ok=True)
                mesh = os.path.join(mesh[0], mesh[1])
            output_mesh = os.path.join(pers_output, mesh)
            if not os.path.exists(output_mesh):
                print("Fitting file: " + output_mesh)
                mesh_path = os.path.join(pers_path, mesh)
                subprocess.call(["python", "COMA_FLAME_indiv.py", "--mesh_path", mesh_path, "--output_mesh", output_mesh,
                                 "--pers_output", pers_output, "--mesh_name", mesh])
            else:
                print("Fitting already done for : " + output_mesh)



if __name__ == '__main__':
    path_COMA = ""
    output_path = ""
    os.makedirs(output_path, exist_ok=True)
    run_COMA(path_COMA, output_path)