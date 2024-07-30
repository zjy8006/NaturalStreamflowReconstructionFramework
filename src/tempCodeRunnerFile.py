import matplotlib.pyplot as plt
plt.figure(figsize=(7.4861,5))
ax11  = plt.subplot2grid((2,3), (0,0), colspan=2,)#ssa
ax12  = plt.subplot2grid((2,3), (0,2), colspan=1,)
ax21  = plt.subplot2grid((2,3), (1,0), colspan=2,)
ax22  = plt.subplot2grid((2,3), (1,2), colspan=1,)
plt.tight_layout()
plt.show()