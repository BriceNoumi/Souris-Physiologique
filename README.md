# Souris-Physiologique
Recherche et innovation

# Usage

1. Install all dependencies
* [NumPy](https://github.com/numpy/numpy)
* [PySerial](https://pypi.org/project/pyserial/3.0/)
* [PyBluez](https://pypi.org/project/PyBluez/)
* [BITalino API](https://github.com/BITalinoWorld/revolution-python-api)

2. Edit `CONFIGURATION.py` according to device

3. Start sampling server: `py server.py`

4. Start client: `mch_phymice.py`

# Wiring

* A1: Accelerometer X-axis
* A2: Accelerometer Y-axis
* A3: Force sensor
* A4: EMG
