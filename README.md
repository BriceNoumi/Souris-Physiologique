# Souris-Physiologique
Recherche et innovation

# Usage

1. Install Python 3.10.11

2. Install all dependencies
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](https://scipy.org/install/)
* [Matplotlib](https://matplotlib.org/stable/users/getting_started/)
* [PySerial](https://pypi.org/project/pyserial/3.0/)
* [PyBluez](https://pypi.org/project/PyBluez/)
* [BITalino API](https://github.com/BITalinoWorld/revolution-python-api)

3. Edit `CONFIGURATION.py` according to device

4. Turn on device and start sampling server: `py server.py`

5. Start client: `py mch_phymice.py`

# Wiring

* A1: Accelerometer X-axis
* A2: Accelerometer Y-axis
* A3: Force sensor
* A4: EMG
