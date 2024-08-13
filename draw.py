import matplotlib.pyplot as plt

loss = []
bleu = []
with open("log.txt") as file:
    info = file.readlines()
    for i in info:
        i = i.split(':')
        loss.append(float(i[2][:-12]))
        bleu.append(float(i[3]))

print("Bleu last:{}, best:{}(Epoch:{})".format(bleu[-1], max(bleu), bleu.index(max(bleu))))

plt.subplot(1, 2, 1)
plt.plot(range(len(loss)), loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(len(loss)), bleu)
plt.xlabel('epochs')
plt.ylabel('bleu')
plt.grid()

plt.tight_layout()
plt.show()