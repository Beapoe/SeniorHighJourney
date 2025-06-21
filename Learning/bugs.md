## 2025.02.10：19：55
    在处理库依赖时要注意Cmakelists中设定的cpp版本，否则即使在源文件中允许了高于这个版本的特性，cmake仍不会采用

## 2025.06.21:23:28
    "labels = (data.norm(1) > 0.5).float().unsqueeze(1)
    IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)"

    data是一个二维张量，尝试取其第二维度向量模长，但忽略了按位传参的问题：如果不指定"dim="那么按位传参传入的参数会被视为第一个参数而非dim
    
    正确代码：
    labels = (data.norm(dim=1) > 0.5).float().unsqueeze(1)