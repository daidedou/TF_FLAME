import os
import six
import numpy as np
import tensorflow as tf
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer

from tf_smpl.batch_smpl import SMPL
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt

def fit_3D_mesh(target_3d_mesh_fname, model_fname, weights, show_fitting=True):
    '''
    Fit FLAME to 3D mesh in correspondence to the FLAME mesh (i.e. same number of vertices, same mesh topology)
    :param target_3d_mesh_fname:    target 3D mesh filename
    :param model_fname:             saved FLAME model
    :param weights:             weights of the individual objective functions
    :return: a mesh with the fitting results
    '''

    target_mesh = Mesh(filename=target_3d_mesh_fname)

    tf_trans = tf.Variable(np.zeros((1,3)), name="trans", dtype=tf.float64, trainable=True)
    tf_rot = tf.Variable(np.zeros((1,3)), name="pose", dtype=tf.float64, trainable=True)
    tf_pose = tf.Variable(np.zeros((1,12)), name="pose", dtype=tf.float64, trainable=True)
    tf_shape = tf.Variable(np.zeros((1,300)), name="shape", dtype=tf.float64, trainable=True)
    tf_exp = tf.Variable(np.zeros((1,100)), name="expression", dtype=tf.float64, trainable=True)
    smpl = SMPL(model_fname)
    tf_model = tf.squeeze(smpl(tf_trans,
                               tf.concat((tf_shape, tf_exp), axis=-1),
                               tf.concat((tf_rot, tf_pose), axis=-1)))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        mesh_dist = tf.reduce_sum(tf.square(tf.subtract(tf_model, target_mesh.v)))
        neck_pose_reg = tf.reduce_sum(tf.square(tf_pose[:,:3]))
        jaw_pose_reg = tf.reduce_sum(tf.square(tf_pose[:,3:6]))
        eyeballs_pose_reg = tf.reduce_sum(tf.square(tf_pose[:,6:]))
        shape_reg = tf.reduce_sum(tf.square(tf_shape))
        exp_reg = tf.reduce_sum(tf.square(tf_exp))

        # Optimize global transformation first
        vars = [tf_trans, tf_rot]
        loss = mesh_dist
        optimizer = scipy_pt(loss=loss, var_list=vars, method='BFGS', options={'disp': 1})
        print('Optimize rigid transformation')
        optimizer.minimize(session)

        # Optimize for the model parameters
        vars = [tf_trans, tf_rot, tf_pose, tf_shape, tf_exp]
        loss = weights['data'] * mesh_dist + weights['shape'] * shape_reg + weights['expr'] * exp_reg + \
               weights['neck_pose'] * neck_pose_reg + weights['jaw_pose'] * jaw_pose_reg + weights['eyeballs_pose'] * eyeballs_pose_reg

        optimizer = scipy_pt(loss=loss, var_list=vars, method='BFGS', options={'disp': 1})
        print('Optimize model parameters')
        optimizer.minimize(session)

        print('Fitting done')

        if show_fitting:
            # Visualize fitting
            mv = MeshViewer()
            fitting_mesh = Mesh(session.run(tf_model), smpl.f)
            fitting_mesh.set_vertex_colors('light sky blue')

            mv.set_static_meshes([target_mesh, fitting_mesh])
            six.moves.input('Press key to continue')

        return Mesh(session.run(tf_model), smpl.f), session.run(tf_pose), session.run(tf_shape), session.run(tf_exp)

def run_COMA(path, output_path):
    # Path of the FLAME model
    model_fname = './models/generic_model.pkl'
    # model_fname = '/models/female_model.pkl'
    # model_fname = '/models/male_model.pkl'

    weights = {}
    # Weight of the data term
    weights['data'] = 1000.0
    # Weight of the shape regularizer (the lower, the less shape is constrained)
    weights['shape'] = 1e-4
    # Weight of the expression regularizer (the lower, the less expression is constrained)
    weights['expr'] = 1e-4
    # Weight of the neck pose (i.e. neck rotationh around the neck) regularizer (the lower, the less neck pose is constrained)
    weights['neck_pose'] = 1e-4
    # Weight of the jaw pose (i.e. jaw rotation for opening the mouth) regularizer (the lower, the less jaw pose is constrained)
    weights['jaw_pose'] = 1e-4
    # Weight of the eyeball pose (i.e. eyeball rotations) regularizer (the lower, the less eyeballs pose is constrained)
    weights['eyeballs_pose'] = 1e-4
    # Show landmark fitting (default: red = target landmarks, blue = fitting landmarks)
    show_fitting = False
    all_pers = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
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
                result_mesh, pose, shape, exp = fit_3D_mesh(mesh_path, model_fname, weights, show_fitting=show_fitting)
                result_mesh.write_ply(output_mesh)
                np_params = {"pose": pose, "shape": shape, "exp": exp}
                np.save(os.path.join(pers_output, mesh.replace(".ply", ".npy")), np_params)
            else:
                print("Fitting already done for : " + output_mesh)



if __name__ == '__main__':
    path_COMA = ""
    output_path = ""
    os.makedirs(output_path, exist_ok=True)
    run_COMA(path_COMA, output_path)