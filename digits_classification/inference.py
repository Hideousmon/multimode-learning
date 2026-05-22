# Licensed under the MIT License. See LICENSE file for details.
# Copyright (c) 2026 Zhenyu Zhao


from splayout import *
import numpy as np


def reload_func():
    global fdtd
    fdtd.eval("clear;")
    fdtd.save("./reload.fsp")
    fdtd = FDTDSimulation(load_file="./reload.fsp")
    fdtd.global_monitor_set_flag = 1
    fdtd.global_source_set_flag = 1
    fdtd.wavelength_start = 1.55e-6
    fdtd.wavelength_end = 1.55e-6
    fdtd.frequency_points = 1


def make_physical_model():
    fdtd = FDTDSimulation()

    base_slab = Waveguide(Point(-44.8 / 2 - 0.45, 0), Point(44.8 / 2 + 0.45, 0), width=20.6,
                          material=perturbed_effSi, z_start=-0.11, z_end=0.11)
    base_slab.draw_on_lumerical_CAD(fdtd)

    # input waveguide
    input_waveguide = Waveguide(Point(-44.8 / 2 - 0.45, 0) - (3, 0), Point(-44.8 / 2 - 0.45, 0), width=20.6,
                                material=effSi, z_start=-0.11, z_end=0.11)
    input_waveguide.draw_on_lumerical_CAD(fdtd)

    # output waveguide
    output_waveguide = Waveguide(Point(44.8 / 2 + 0.45, 0) + (3, 0), Point(44.8 / 2 + 0.45, 0), width=20.6,
                                 material=effSi, z_start=-0.11, z_end=0.11)
    output_waveguide.draw_on_lumerical_CAD(fdtd)

    fdtd.add_fdtd_region(bottom_left_corner_point=Point(-22.4 - 1.8, -12.5),
                         top_right_corner_point=Point(22.4 + 1.8, 12.5),
                         dimension=2, simulation_time=5000, height=0.8, z_symmetric=0, use_gpu=0)
    fdtd.eval("set(\"mesh refinement\", \"volume average\");")

    # input sources
    for i in range(0, 65):
        fdtd.add_mode_source(Point(-22.4 - 1.5, 0), width=22, source_name="Forward_" + str(i),
                             direction=FORWARD, mode_number=2 + i * 2,
                             wavelength_start=1.55, wavelength_end=1.55)

    # output monitors
    for i in range(0, 10):
        fdtd.add_mode_expansion(Point(22.4 + 1.5, 0), mode_list=[2 + i * 2], width=22,
                                expansion_name="Monitor_" + str(i),
                                points=1, auto_update=0)

    fdtd.add_mesh_region(Point(-22.4 - 1.8, -12.5),
                         Point(22.4 + 1.8, 12.5), x_mesh=0.025, y_mesh=0.025, z_mesh=0.02,
                         height=0.22)

    return fdtd


if __name__ == '__main__':
    np.random.seed(5)
    effSi = 2.85189
    perturbed_effSi = 2.53532
    fdtd = make_physical_model()

    # import structure
    fdtd.add_structure_from_gdsii("./digits.gds", cellname="digits", layer=1, datatype=0,
                                  material=effSi, z_start=-0.11, z_end=0.11)

    S = []
    for i in range(0, 65):
        # reload
        if i%10 == 0 and i!=0:
            reload_func()
        disable_list = []
        for j in range(0, 65):
            if j == i:
                fdtd.set_enable("Forward_" + str(j))
            else:
                disable_list.append("Forward_" + str(j))
        fdtd.set_disable(disable_list)
        fdtd.run()
        S_x0 = []
        wl = fdtd.get_wavelength()
        source_power = fdtd.get_source_power("Forward_"+str(i), wl)
        for j in range(0, 10):
            mode_exp_data_set = fdtd.fdtd.getresult("Monitor_"+str(j), 'expansion for Output')
            fwd_trans_coeff = mode_exp_data_set['a'] * np.sqrt(mode_exp_data_set['N'].real)
            s_j0 = fwd_trans_coeff.flatten() / np.sqrt(source_power)
            S_x0.append(s_j0)

        S.append(S_x0)
        fdtd.switch_to_layout()

    S = np.array(S).transpose((1, 0, 2))
    transfer_func = S[:, :, 0]


    train_data_features = np.load("../datasets/ocr/train_data_features.npy")
    train_data_features = np.concatenate([train_data_features, np.zeros_like(train_data_features[:, 0:1])], axis=1)
    train_data_targets = np.load("../datasets/ocr/train_data_targets.npy")
    test_data_features = np.load("../datasets/ocr/test_data_features.npy")
    test_data_features = np.concatenate([test_data_features, np.zeros_like(test_data_features[:, 0:1])], axis=1)
    test_data_targets = np.load("../datasets/ocr/test_data_targets.npy")

    # inference on test dataset
    transfer_func_expand = np.repeat(np.expand_dims(transfer_func, axis=0), test_data_features.shape[0], axis=0)
    feature_data = np.exp(-1j * np.pi * np.expand_dims(test_data_features, axis=(2)))
    transmission = np.power(
        np.abs(np.matmul(
            transfer_func_expand, feature_data)), 2)[:, :,
                   0] / 65  # normalization for input

    # calculate loss on test dataset
    nmse_loss_test = np.mean(
        np.sum(np.power(transmission / np.expand_dims(np.sum(transmission, axis=1), 1) - test_data_targets, 2), axis=1))

    # calculate accuracy test dataset
    argmax_fom = np.argmax(transmission, axis=1)
    argmax_target = np.argmax(test_data_targets, axis=1)
    match_points = argmax_fom == argmax_target
    acc_test = np.sum(match_points) / transmission.shape[0]

    # inference on train dataset
    transfer_func_expand = np.repeat(np.expand_dims(transfer_func, axis=0), train_data_features.shape[0], axis=0)

    feature_data = np.exp(-1j * np.pi * np.expand_dims(train_data_features, axis=(2)))
    transmission = np.power(
        np.abs(np.matmul(transfer_func_expand, feature_data)), 2)[:, :,
                   0] / 65  # normalization for input

    # calculate loss on train dataset
    nmse_loss_train = np.mean(
        np.sum(np.power(transmission / np.expand_dims(np.sum(transmission, axis=1), 1) - train_data_targets, 2), axis=1))

    # calculate accuracy on train dataset
    argmax_fom = np.argmax(transmission, axis=1)
    argmax_target = np.argmax(train_data_targets, axis=1)
    match_points = argmax_fom == argmax_target
    acc_train = np.sum(match_points) / transmission.shape[0]

    print("test accuracy: ", acc_test, " , test loss: ", nmse_loss_test, " , train accuracy: ",
          acc_train, " , train loss: ", nmse_loss_train)


