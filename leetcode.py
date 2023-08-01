	#pytorch封装卷积层
class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule,self).__init__()
        #定义六层卷积层
        #两层HDC（1,2,5,1,2,5）
        self.conv = nn.Sequential(
            #第一层 (3-1)*1+1=3 （64-3)/1 + 1 =62
            nn.Conv1d(in_channels = 3,out_channels = 32,kernel_size = 3 , stride = 1,padding=0,dilation=1),
            nn.BatchNorm1d(32),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            #第二层 (3-1)*2+1=5 （62-5)/1 + 1 =58
            nn.Conv1d(in_channels = 32,out_channels = 32,kernel_size = 3 , stride = 1,padding=0,dilation=2),
            nn.BatchNorm1d(32),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            #第三层 (3-1)*5+1=11  (58-11)/1 +1=48
            nn.Conv1d(in_channels = 32,out_channels = 64,kernel_size = 3 , stride = 1,padding=0,dilation=5),
            nn.BatchNorm1d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
             #第四层(3-1)*1+1=3 （48-3)/1 + 1 =46
            nn.Conv1d(in_channels = 64,out_channels = 64,kernel_size = 3 , stride = 1,padding=0,dilation=1),
            nn.BatchNorm1d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            #第五层 (3-1)*2+1=5 （46-5)/1 + 1 =42
            nn.Conv1d(in_channels = 64,out_channels = 64,kernel_size = 3 , stride = 1,padding=0,dilation=2),
            nn.BatchNorm1d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            #第六层 (3-1)*5+1=11  (42-11)/1 +1=32
            nn.Conv1d(in_channels = 64,out_channels = 128,kernel_size = 3 , stride = 1,padding=0,dilation=5),
            nn.BatchNorm1d(128),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True)

        )
        #输出层,将通道数变为分类数量
        self.fc = nn.Linear(128,num_classes)

    def forward(self,x):
        #图片经过三层卷积，输出维度变为(batch_size,C_out,H,W)
        out = self.conv(x)
        #使用平均池化层将图片的大小变为1x1,第二个参数为最后输出的长和宽（这里默认相等了）
        out = F.avg_pool1d(out,32)
        #将张量out从shape batchx128x1x1 变为 batch x128
        out = out.squeeze()
        #输入到全连接层将输出的维度变为3
        out = self.fc(out)
        return out

