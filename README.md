# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```

**mnist.txt Sample Output**

Epoch 1 loss 3.5565916852168886 valid acc 13/16
Epoch 1 loss 5.495344715556503 valid acc 15/16
Epoch 1 loss 3.370880391701295 valid acc 15/16
Epoch 1 loss 3.5816906345559723 valid acc 11/16
Epoch 1 loss 2.952459611037555 valid acc 12/16
Epoch 1 loss 3.7309826915466004 valid acc 12/16
Epoch 1 loss 3.050192402871556 valid acc 13/16
Epoch 1 loss 1.7996800980752965 valid acc 15/16
Epoch 1 loss 3.362488675245767 valid acc 13/16
Epoch 1 loss 5.728345728256626 valid acc 15/16
Epoch 1 loss 3.047956248437136 valid acc 14/16
Epoch 1 loss 2.0797772144748174 valid acc 14/16
Epoch 1 loss 3.2346218354003735 valid acc 16/16
Epoch 1 loss 2.3698475678481756 valid acc 13/16
Epoch 1 loss 3.3073407627918 valid acc 14/16
Epoch 1 loss 2.1597382523746065 valid acc 16/16
Epoch 1 loss 3.029073888907315 valid acc 16/16
Epoch 1 loss 2.203027203711712 valid acc 13/16
Epoch 1 loss 1.572887134154193 valid acc 13/16
Epoch 1 loss 3.313460209336477 valid acc 16/16
Epoch 1 loss 1.7571441476635639 valid acc 16/16
Epoch 1 loss 1.9357346352761247 valid acc 14/16
Epoch 1 loss 2.541300770032917 valid acc 15/16
Epoch 1 loss 2.3140871313416636 valid acc 16/16
Epoch 1 loss 1.8116188891095104 valid acc 16/16
Epoch 1 loss 2.1531047603490143 valid acc 15/16
Epoch 1 loss 2.2135342676653424 valid acc 15/16
Epoch 1 loss 1.9625870736075515 valid acc 15/16
Epoch 1 loss 2.3063034903948845 valid acc 14/16
Epoch 1 loss 2.7778263224952897 valid acc 15/16
Epoch 1 loss 1.8597230750083142 valid acc 15/16
Epoch 1 loss 2.0343942896229192 valid acc 15/16
Epoch 1 loss 2.1235248029514073 valid acc 14/16
Epoch 1 loss 2.273491361113812 valid acc 15/16
Epoch 1 loss 2.15744756575824 valid acc 14/16
Epoch 1 loss 2.4033779845250214 valid acc 15/16
Epoch 1 loss 2.0785169540169712 valid acc 15/16
Epoch 1 loss 3.16208988649025 valid acc 15/16
Epoch 2 loss 0.41458037144631177 valid acc 16/16
Epoch 2 loss 1.5895905561079147 valid acc 15/16
Epoch 2 loss 2.216182833840646 valid acc 15/16
Epoch 2 loss 2.0711735518978083 valid acc 14/16
Epoch 2 loss 1.5841658697120544 valid acc 14/16
Epoch 2 loss 1.5870016903836308 valid acc 15/16
Epoch 2 loss 1.5308747219027525 valid acc 14/16
Epoch 2 loss 2.642641104478546 valid acc 15/16
Epoch 2 loss 1.9001728813689782 valid acc 14/16
Epoch 2 loss 1.3892404898544446 valid acc 14/16
Epoch 2 loss 1.8216679471539177 valid acc 16/16
Epoch 2 loss 2.7958307537689144 valid acc 16/16
Epoch 2 loss 2.7815342697222176 valid acc 15/16
Epoch 2 loss 3.092883562425727 valid acc 14/16
Epoch 2 loss 3.1280847993024095 valid acc 14/16
Epoch 2 loss 1.2202969491038613 valid acc 14/16
Epoch 2 loss 2.322542514230594 valid acc 14/16
Epoch 2 loss 2.4239302149772612 valid acc 15/16
Epoch 2 loss 1.403436547925129 valid acc 14/16
Epoch 2 loss 1.7020004360763357 valid acc 15/16
Epoch 2 loss 1.7145655588022797 valid acc 14/16
Epoch 2 loss 1.6496635979532008 valid acc 15/16
Epoch 2 loss 0.5043420733351484 valid acc 14/16
Epoch 2 loss 1.294443905733941 valid acc 14/16
Epoch 2 loss 1.1705860223778808 valid acc 14/16
Epoch 2 loss 1.316486106639924 valid acc 15/16
Epoch 2 loss 1.0217625588219372 valid acc 15/16
Epoch 2 loss 2.8661063444365 valid acc 16/16
Epoch 2 loss 0.9801120692035252 valid acc 16/16
Epoch 2 loss 0.5571357363426367 valid acc 16/16
Epoch 2 loss 1.9024376770145204 valid acc 16/16
Epoch 2 loss 1.8364787758026035 valid acc 14/16
Epoch 2 loss 0.6740000199582562 valid acc 15/16
Epoch 2 loss 0.8034608968008747 valid acc 16/16
Epoch 2 loss 2.67459789689602 valid acc 16/16
Epoch 2 loss 1.5002011184592128 valid acc 14/16
Epoch 2 loss 1.2806487013519139 valid acc 15/16
Epoch 2 loss 1.1890865457427335 valid acc 16/16
Epoch 2 loss 1.2696528582634352 valid acc 15/16
Epoch 2 loss 1.1657344122687163 valid acc 14/16
Epoch 2 loss 1.054596225332485 valid acc 15/16
Epoch 2 loss 1.6642904536981649 valid acc 16/16
Epoch 2 loss 0.7367662298232711 valid acc 16/16
Epoch 2 loss 0.9583367773476196 valid acc 16/16
Epoch 2 loss 2.163934865757467 valid acc 16/16
Epoch 2 loss 0.8242470334926633 valid acc 15/16
Epoch 2 loss 1.1485327988722625 valid acc 16/16
Epoch 2 loss 1.924495114477968 valid acc 15/16
Epoch 2 loss 0.821581778085802 valid acc 15/16
Epoch 2 loss 1.086554898906355 valid acc 15/16
Epoch 2 loss 1.040783087577421 valid acc 14/16
Epoch 2 loss 1.4039953988768121 valid acc 16/16
Epoch 2 loss 1.4042974591991648 valid acc 15/16
Epoch 2 loss 1.3007288058983932 valid acc 16/16
Epoch 2 loss 1.1928491222949178 valid acc 15/16
Epoch 2 loss 0.7820991092117455 valid acc 16/16
Epoch 2 loss 0.7578993732367876 valid acc 16/16
Epoch 2 loss 2.2070699510992977 valid acc 15/16
Epoch 2 loss 1.4142582654799711 valid acc 14/16
Epoch 2 loss 1.0569837343015605 valid acc 15/16
Epoch 2 loss 1.6621858919467525 valid acc 15/16
Epoch 2 loss 1.2636483662166662 valid acc 16/16
Epoch 2 loss 1.7174141562974987 valid acc 15/16
Epoch 3 loss 0.0817978201970452 valid acc 16/16
Epoch 3 loss 0.7376758404681298 valid acc 16/16
Epoch 3 loss 1.7854894565199273 valid acc 16/16
Epoch 3 loss 1.0909100756515404 valid acc 15/16
Epoch 3 loss 0.6793600833308666 valid acc 15/16
Epoch 3 loss 0.6233723187872695 valid acc 16/16
Epoch 3 loss 1.5450457421388253 valid acc 15/16
Epoch 3 loss 1.5147668615056267 valid acc 15/16
Epoch 3 loss 1.1890404834397956 valid acc 16/16
Epoch 3 loss 0.9958225635879797 valid acc 15/16
Epoch 3 loss 0.6387889882675304 valid acc 15/16
Epoch 3 loss 1.813401079809603 valid acc 15/16
Epoch 3 loss 2.7892839778167677 valid acc 15/16


**sentiment.txt Sample Output**

Epoch 23, loss 22.172777313696912, train accuracy: 78.67%
	Validation accuracy: 68.00%
	Best Valid accuracy: 74.00%
Epoch 24, loss 21.779342135403876, train accuracy: 76.22%
	Validation accuracy: 74.00%
	Best Valid accuracy: 74.00%
Epoch 25, loss 21.168984553181982, train accuracy: 78.00%
	Validation accuracy: 74.00%
	Best Valid accuracy: 74.00%
Epoch 26, loss 20.480713251416724, train accuracy: 76.89%
	Validation accuracy: 75.00%
	Best Valid accuracy: 75.00%
Epoch 27, loss 19.96623144420461, train accuracy: 78.22%
	Validation accuracy: 70.00%
	Best Valid accuracy: 75.00%
Epoch 28, loss 19.46619131707657, train accuracy: 80.44%
	Validation accuracy: 73.00%
	Best Valid accuracy: 75.00%


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py

