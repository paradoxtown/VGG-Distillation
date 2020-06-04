import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['axes.unicode_minus'] = False

bl_data = \
"Accuracy of plane : 87.6106 %\
Accuracy of   car : 97.3913 %\
Accuracy of  bird : 89.4737 %\
Accuracy of   cat : 74.4186 %\
Accuracy of  deer : 86.3636 %\
Accuracy of   dog : 92.0354 %\
Accuracy of  frog : 91.7355 %\
Accuracy of horse : 91.4062 %\
Accuracy of  ship : 95.5752 %\
Accuracy of truck : 94.1176 %"


kd_data =\
"Accuracy of plane : 93.8053 %\
Accuracy of   car : 99.1304 %\
Accuracy of  bird : 89.4737 %\
Accuracy of   cat : 87.5969 %\
Accuracy of  deer : 95.4545 %\
Accuracy of   dog : 92.9204 %\
Accuracy of  frog : 91.7355 %\
Accuracy of horse : 96.8750 %\
Accuracy of  ship : 96.4602 %\
Accuracy of truck : 97.0588 %"


def parse(text):
    return [round(float(digit.split(':')[1][:-1].strip()), 1) for digit in text.split('%')[:-1]]


def draw_bar():
    labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    bl = parse(bl_data)
    kd = parse(kd_data)

    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, bl, width, label='student-A')
    rects2 = ax.bar(x + width/2, kd, width, label='teacher')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    # ax.set_title('Accuracy Comparison of Each Class Before and After SP Knowledge Distillation(%)')
    ax.set_title('Accuracy Comparison of Each Class Between Teacher and Student-A(%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc=4)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    # fig.tight_layout()
    plt.ylim((70, 100))
    plt.style.use('fivethirtyeight')
    plt.show()
    fig.savefig('./img/T-SP-A-4.png', dpi=600)


def draw_chart():
    fig, ax = plt.subplots()
    ax.set_title('Different Blocks\' SP Knowledge Distillation(%)', fontsize=16)
    plt.plot(['1', '2', '3', '4', '5'], [90.40, 90.85, 91.66, 91.70, 90.88])
    plt.xlabel('Max Pooling Layer Number', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.show()
    fig.savefig('./img/SP-A-L.png', dpi=600)


def draw_diff():
    labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    bl = parse(bl_data)
    kd = parse(kd_data)
    diff = []
    for i in range(10):
        diff.append(round(float(kd[i] - bl[i]), 2))

    x = np.arange(len(labels))  # the label locations
    width = 0.8  # the width of the bars

    fig, ax = plt.subplots()
    rects = ax.bar(x, diff, width, label='Difference', facecolor='#B0C4DE')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_title('Accuracy Difference of AT (%)', fontsize=14)
    # ax.set_title('Accuracy Comparison of Each Class Between Teacher and Student-B ')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    ax.legend(fontsize=16)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=16)

    autolabel(rects)
    # fig.tight_layout()
    plt.style.use('fivethirtyeight')
    plt.ylim((-10, 10))
    plt.show()
    fig.savefig('./img/Diff-A-AT.png', dpi=600)


draw_bar()
# draw_chart()
# draw_diff()
