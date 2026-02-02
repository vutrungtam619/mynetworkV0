import numpy as np
import matplotlib.pyplot as plt

# Giả sử diện tích bounding box A_b = 1
A_b = 0.36

# N_b chạy từ 1 đến 200
N_b = np.arange(1, 1001)

# Tính mật độ điểm p_b và giá trị log nén
p_b = N_b / A_b
log_p_b = np.log(1 + p_b)

# Vẽ biểu đồ
plt.figure()
plt.plot(N_b, log_p_b)
plt.xlabel(r'$N_b$ (Number of points)')
plt.ylabel(r'$\log(1 + p_b)$')
plt.grid(True)

plt.show()
