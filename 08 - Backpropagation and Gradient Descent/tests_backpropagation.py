import torch
from torch import nn, optim
from torchvision import datasets, transforms


class MyNet(nn.Module):
    def __init__(self, n_l=[2, 3, 2]):
        super().__init__()

        # number of layers in our network (following Andrew's notations)
        self.L = len(n_l) - 1
        self.n_l = n_l

        # Where we will store our neuron values
        # - z: before activation function
        # - a: after activation function (a=f(z))
        self.z = {i: None for i in range(1, self.L + 1)}
        self.a = {i: None for i in range(self.L + 1)}

        # Where we will store the gradients for our custom backpropagation algo
        self.dL_dw = {i: None for i in range(1, self.L + 1)}
        self.dL_db = {i: None for i in range(1, self.L + 1)}

        # Our activation functions
        self.f = {i: lambda x: torch.tanh(x) for i in range(1, self.L + 1)}

        # Derivatives of our activation functions
        self.df = {
            i: lambda x: (1 / (torch.cosh(x) ** 2)) for i in range(1, self.L + 1)
        }

        # fully connected layers
        # We have to use nn.ModuleDict and to use strings as keys here to
        # respect pytorch requirements (otherwise, the model does not learn)
        self.fc = nn.ModuleDict({str(i): None for i in range(1, self.L + 1)})
        for i in range(1, self.L + 1):
            self.fc[str(i)] = nn.Linear(in_features=n_l[i - 1], out_features=n_l[i])

    def forward(self, x):
        # Input layer
        self.a[0] = torch.flatten(x, 1)

        # Hidden layers until output layer
        for i in range(1, self.L + 1):

            # fully connected layer
            self.z[i] = self.fc[str(i)](self.a[i - 1])
            # activation
            self.a[i] = self.f[i](self.z[i])

        # return output
        return self.a[self.L]


def grad_check(model, x, y, loss_fn, eps=10e-5):
    """
    Compare our gradients with finite differences computations
    """
    grad_approx = []
    grad_backprop = []
    flag = True

    with torch.no_grad():

        # Go through all layers
        for i_layer in range(1, model.L + 1):
            # Each layer has a weight and bias parameter
            for param_name in ["weight", "bias"]:

                param = getattr(model.fc[str(i_layer)], param_name)
                if len(param.data.shape) == 1:
                    j_range = 1
                else:
                    j_range = param.data.shape[1]

                for i in range(param.data.shape[0]):
                    for j in range(j_range):

                        if len(param.data.shape) == 1:
                            idx = i
                        else:
                            idx = (i, j)

                        # Compute loss with param += eps
                        model1 = MyNet(model.n_l)
                        model1.load_state_dict(model.state_dict())
                        param1 = getattr(model1.fc[str(i_layer)], param_name)
                        param1.data[idx] = param.data[idx] + eps
                        loss1 = loss_fn(model1(x), y).sum().item()

                        # Compute loss with param -= eps
                        model2 = MyNet(model.n_l)
                        model2.load_state_dict(model.state_dict())
                        param2 = getattr(model2.fc[str(i_layer)], param_name)
                        param2.data[idx] = param.data[idx] - eps
                        loss2 = loss_fn(model2(x), y).sum().item()

                        # Gradients computed with backprop and autograd
                        grad_approx.append((loss1 - loss2) / (2 * eps))
                        if param_name == "bias":
                            grad_backprop.append(model.dL_db[i_layer][idx])
                        else:
                            grad_backprop.append(model.dL_dw[i_layer][idx])

        g_approx = torch.tensor(grad_approx)
        g_backprop = torch.tensor(grad_backprop)

        res_backprop = torch.norm(g_approx - g_backprop) / (
            torch.norm(g_approx) + torch.norm(g_backprop)
        )
        if res_backprop > eps:
            flag = False
    return flag, res_backprop


def load_MNIST(data_path="../data/", preprocessor=None):

    if preprocessor is None:
        preprocessor = transforms.Compose(
            [
                transforms.CenterCrop(24),
                transforms.ToTensor(),
                transforms.Normalize(0.1306, 0.3080),
            ]
        )

    # load just test dataset because it's smaller
    data_test = datasets.MNIST(
        data_path, train=False, download=True, transform=preprocessor
    )

    return data_test


def check_computational_graph(model):
    """
    Make sure all trainable parameters require grad and are leaves
    """
    flag = True
    # Go through all layers
    for i_layer in range(1, model.L + 1):
        # Each layer has a weight and bias parameter
        for param_name in ["weight", "bias"]:

            # 'getattr(object, string variable)' is like `object.myattribute` when variable = "myattribute"
            param = getattr(model.fc[str(i_layer)], param_name)
            msg = " !!!! WARNING !!!!\nmodel.fc[" + str(i_layer) + "]." + param_name
            if not param.requires_grad:
                print(msg + " does not require grad!")
                print(param)
                res = False
            if not param.is_leaf:
                print(msg + " is not a leaf!")
                print(param)
                flag = False
    return flag


def relative_error(a, b):
    return torch.norm(a - b) / torch.norm(a)


def compare_with_autograd(model, verbose=False):
    """
    Compare our gradients with autograd's computations
    """
    flag = True
    if verbose:
        print("\n --------- Comparing with autograd values  ----------- ")
        print("\n ******* fc['1'].weight.grad ******* ")
        print("  Our computation:\n", model.dL_dw[1])
        print("\n  Autograd's computation:\n", model.fc["1"].weight.grad)
        print("\n ********* fc['1'].bias.grad ******* ")
        print("  Our computation:\n", model.dL_db[1])
        print("  Autograd's computation:\n", model.fc["1"].bias.grad)
    msg = "\n ------------------- relative error ------------------ \n"
    for i in range(1, model.L + 1):
        err = relative_error(model.fc[str(i)].weight.grad, model.dL_dw[i])
        if err > 0.001 or verbose:
            print(
                msg
                + "(fc["
                + str(i)
                + "].weight.grad, model.dL_dw["
                + str(i)
                + "]):   %.4f" % err
            )
            flag &= not (err > 0.001)
            msg = ""

        err = relative_error(model.fc[str(i)].bias.grad, model.dL_db[i])
        if err > 0.001 or verbose:
            print(
                msg
                + "(fc["
                + str(i)
                + "].bias.grad,   model.dL_db["
                + str(i)
                + "]):   %.4f" % err
            )
            flag &= not (err > 0.001)
            msg = ""
    return flag


def check_gradients(
    model,
    backprop_fn,
    optimizer,
    loss_fn,
    x,
    y_true,
    n_epochs=5,
    eps=10e-5,
    verbose=False,
):
    """
    Training loop that compares our gradients with autograd's computations
    """
    model.train()
    flag_autograd = True
    flag_gradcheck = True

    for epoch in range(1, n_epochs + 1):

        gradcheck_ok = True
        autograd_ok = True
        res_backprop = []

        for i, (input, out_expected) in enumerate(zip(x, y_true)):

            out = model(input)
            loss = loss_fn(out, out_expected)

            loss.sum().backward()

            backprop_fn(model, out_expected, out)
            flag, res = grad_check(model, input, out_expected, loss_fn, eps=eps)
            gradcheck_ok &= flag
            res_backprop.append(round(res.item(), 4))

            optimizer.step()

            if i == (len(x) - 1) and (epoch % 1 == 0):
                if verbose or not gradcheck_ok:
                    print(
                        "\n ====================== Epoch %d ====================== "
                        % epoch
                    )
                    print("\n -------- Gradcheck with finite differences  --------- ")
                    print(" residual error:\n", res_backprop)

                autograd_ok &= compare_with_autograd(model, verbose=verbose)

            optimizer.zero_grad()

        flag_gradcheck &= gradcheck_ok
        flag_autograd &= autograd_ok

    return flag_autograd, flag_gradcheck


def main_test(backprop_fn, model, eps=10e-5, verbose=False, data="toy"):
    """
    Main test function

    Test "backpropagation" by:
    - comparing computed gradients with autograd's computations
    - comparing computed gradients with finite differences computations
    - checking that the weights have been updated
    - checking that all trainable parameters are still attached to
    the computational graph.
    """
    fc1_w_init = model.fc["1"].weight.data
    fc1_b_init = model.fc["1"].bias.data

    # Using a few images of the MNIST dataset
    if data == "mnist":
        if model.n_l[0] != 24 * 24:
            print(
                "Please choose 576 (=24x24) as input dimension if using MNIST dataset"
            )
        data = load_MNIST()
        N = 5
        inputs = [torch.unsqueeze(img, 0).to(dtype=torch.double) for img, _ in data][:N]
        outputs = [torch.Tensor([[label]]) for _, label in data][:N]
        n_epochs = 1
    # Create toy dataset
    else:
        N = 10
        inputs = torch.stack([i + torch.randn((1, model.n_l[0])) for i in range(N)])
        outputs = torch.stack(
            [
                torch.Tensor([[(-1) ** j * i for j in range(model.n_l[-1])]])
                for i in range(N)
            ]
        )
        n_epochs = 5

    print("\n __________________________________________________________________ ")
    print("                          Check gradients                             ")
    print(" __________________________________________________________________ ")
    loss_fn = nn.MSELoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # Training loop that compares our gradients with autograd's computations
    autograd_ok, gradcheck_ok = check_gradients(
        model,
        backprop_fn,
        optimizer,
        loss_fn,
        inputs,
        outputs,
        eps=eps,
        n_epochs=n_epochs,
        verbose=verbose,
    )
    if autograd_ok:
        print("\n TEST PASSED: Gradients consistent with autograd's computations.")
    else:
        print("\n TEST FAILED: Gradients NOT consistent with autograd's computations.")
    if gradcheck_ok:
        print(
            "\n TEST PASSED: Gradients consistent with finite differences computations."
        )
    else:
        print(
            "\n TEST FAILED: Gradients NOT consistent with finite differences computations."
        )

    print("\n __________________________________________________________________ ")
    print("                 Check that weights have been updated               ")
    print(" __________________________________________________________________ ")
    update_weight_ok = True
    update_weight_ok &= not torch.all((fc1_w_init - model.fc["1"].weight.data).bool())
    update_weight_ok &= not torch.all((fc1_b_init - model.fc["1"].bias.data).bool())

    if verbose:
        print(model.fc["1"].weight.data)
        print(model.fc["1"].bias.data)
    if update_weight_ok:
        print("\n TEST PASSED: Weights have been updated.")
    else:
        print("\n TEST FAILED: Weights have NOT been updated.")

    print("\n __________________________________________________________________ ")
    print("                      Check computational graph                     ")
    print(" __________________________________________________________________ ")
    graph_ok = check_computational_graph(model)
    if graph_ok:
        print(
            "\n TEST PASSED: All parameters seem correctly attached to the computational graph!"
        )
    else:
        print(
            "\n TEST FAILED: Some parameters are NOT correctly attached to the computational graph!"
        )
    print("\n __________________________________________________________________ ")
    print("                             Conclusion                     ")
    print(" __________________________________________________________________ ")
    tests = [autograd_ok, gradcheck_ok, update_weight_ok, graph_ok]
    n_tests = len(tests)
    n_passed = sum(tests)
    if n_tests == n_passed:
        print("\n %d / %d: ALL TEST PASSED :)" % (n_passed, n_tests))
    else:
        print(
            "\n %d / %d: SOME TESTS FAILED, use 'verbose=True' and check the output for more details"
            % (n_passed, n_tests)
        )
