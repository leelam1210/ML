import pandas as pd
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix #ghi ma trận nhầm lẫn cho Mô hình Gini
import seaborn as sns

data = pd.read_csv("fish.csv") # Đọc dữ liệu

print(data.head()) #5 bản ghi dữ liệu đầu tiên
print("============================***===============================")

print(data.info()) #thông tin dữ liệu đầu vào

print("============================***===============================")
# data = data.drop(["Unnamed: 0"], axis=1) #Loại bỏ cột chỉ mục đầu tiên

print("============================***===============================")
#mô tả thuộc tính dữ liệu các cột số
pd.set_option("display.float_format", "{:.2f}".format)
print(data.describe())
print(data.shape)

print("============================***===============================")
print("Nhận danh sách các biến phân loại:") # Nhận danh sách các biến phân loại
categorical_col = []
for column in data.columns:
    if data[column].dtype == object and len(data[column].unique()) <= 50:
        categorical_col.append(column)
        print(f"Các giá trị có trong cột: {column} : {data[column].unique()}")
        print(f"Tổng số lượng các loài khác nhau trong 1 thuộc tính: {column} :\n{data[column].value_counts()}")
        print("====================================")

print("============================***===============================")
# data['carat'] = data.carat.astype("category").cat.codes
# print(data.carat.value_counts())

print("DỮ LIỆU SAU KHI ĐƯỢC CHUYỂN ĐỔI VỀ KIỂU SỐ:")
label_data = data.copy() # Tạo bản sao để tránh thay đổi dữ liệu gốc
# Áp dụng bộ mã hóa nhãn cho từng cột với dữ liệu phân loại & chỉnh sửa các giá trị kiểu chuỗi về kiểu số
label_encoder = LabelEncoder()
for col in categorical_col:
    label_data[col] = label_encoder.fit_transform(label_data[col])
print(label_data.head())

#################
# Biểu đồ trực quan hóa dữ liệu tổng quan cho mọi tính năng
data.hist(edgecolor='black', linewidth=0.6, figsize=(20, 20))
plt.show()

# Vẽ biểu đồ tương quan của mọi tính năng với "mục tiêu"
data['Species'] = data.Species.astype("category").cat.codes
sns.set(font_scale=1.2)
plt.figure(figsize=(30, 30))
for i, column in enumerate(categorical_col, 1):
    plt.subplot(3, 3, i)
    g = sns.barplot(x=f"{column}", y='Species', data=data)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.ylabel('Số Lượng Model')
    plt.xlabel(f'{column}')
    plt.show()

# Biểu đồ phân loại theo màu
# data.Species.value_counts()
# pd.crosstab(data.Species,data.Weight).plot(kind="bar",figsize=(20,6))
# plt.title('Heart Disease Frequency for Ages')
# plt.xlabel('Cá theo cân nặng')
# plt.ylabel('Tần số')
# plt.savefig('tansuat.png')
# plt.show()
###

print("============================***===============================")
X= label_data.drop(["Species"], axis = 1)  # X sẽ giữ tất cả các thuộc tính còn lại
y= label_data["Species"]  # y lấy mục tiêu phân tích dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=0)
print(X.shape) # Kích thước của dữ liệu đầu vào
print(y.shape) # Kích thước của dữ liệu đầu ra

features = list(label_data.columns)
features.remove("Species")
#  Dự đoán bằng cách sử dụng cả hai trình phân loại
classifier1 = DecisionTreeClassifier(criterion='gini')
classifier1.fit(X_train, y_train)
classifier2 = DecisionTreeClassifier(criterion='entropy')
classifier2.fit(X_train, y_train)

y_pred_1 = classifier1.predict(X_test)
print(y_pred_1)
y_pred_2 = classifier2.predict(X_test)
print(y_pred_2)

# Tính chính xác của mô hình
acc_1 = accuracy_score(y_test,y_pred_1)
print("Độ chính xác cho mô hình Gini {0:.2f} %".format(acc_1*100))
acc_2 = accuracy_score(y_test,y_pred_2)
print("Độ chính xác cho mô hình Entropy {0:.2f} %".format(acc_2*100))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_1)}")
print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, y_pred_1)}")

# Hàm vẽ cây:
def draw_trees(tree, feature_names, png_file_to_save):
    tree_str = export_graphviz(
        tree, feature_names=features, filled=True, out_file=None
    )
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(png_file_to_save)
    img = pltimg.imread(png_file_to_save)
    imgplot = plt.imshow(img)
    plt.show()

# dtree = DecisionTreeClassifier(criterion = 'entropy')
# dtree.fit(X_train, y_train)
# Gọi hàm để vẽ 1 cây
draw_trees(tree=classifier1, feature_names=features, png_file_to_save="tree_gini.png")
draw_trees(tree=classifier2, feature_names=features, png_file_to_save="tree_emtropy.png")


print("-----END-----")