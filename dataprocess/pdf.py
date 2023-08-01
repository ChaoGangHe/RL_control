import pdfplumber
from openpyxl import Workbook

dicts={"大赢":0,"小赢":0,"大输":0,"小输":0}
class PDF(object):
    def __init__(self, file_path):
        self.pdf_path = file_path
        # 读取pdf文件
        try:
            self.pdf_info = pdfplumber.open(self.pdf_path)
            print('读取文件完成！')
        except Exception as e:
            print('读取文件失败：', e)

    # 打印pdf的基本信息、返回字典，作者、创建时间、修改时间/总页数
    def get_pdf(self):
        pdf_info = self.pdf_info.metadata
        pdf_page = len(self.pdf_info.pages)
        print('pdf共%s页' % pdf_page)
        print("pdf文件基本信息：\n", pdf_info)
        self.close_pdf()

    # 提取表格数据,并保存到excel中
    def get_table(self):
        wb = Workbook()  # 实例化一个工作簿对象
        ws = wb.active  # 获取第一个sheet
        con = 0
        try:
            # 获取每一页的表格中的文字，返回table、row、cell格式：[[[row1],[row2]]]
            for page in self.pdf_info.pages:
                for table in page.extract_tables():
                    for row in table:
                        # 对每个单元格的字符进行简单清洗处理

                        if len(row)>5:
                            #print(row[2])
                            #row_list = [cell.replace('\n', ' ') if cell else '' for cell in row]
                            #ws.append(row_list)  # 写入数据
                            #if len(row_list)>4:
                            #if(row_list.size()>6)
                            row_list = [cell.replace('\n', ' ') if cell else '' for cell in row]
                            #print(row_list[2].replace('/ ', ''), row_list[6])
                            if (row_list[2].find("⾜球 /⼤⼩盘 ⼩")!=-1) and row_list[6].find("赢")!=-1:
                                dicts["小赢"] = dicts["小赢"]+1
                            elif row_list[2].find("⾜球 /上半场⼤⼩盘 ⼩")!=-1 and row_list[6].find("输")!=-1:
                                dicts["小输"] = dicts["小输"] + 1
                            elif row_list[2].find("⾜球 /上半场⼤⼩盘 ⼩")!=-1 and row_list[6].find("赢")!=-1:
                                dicts["小赢"] = dicts["小赢"] + 1
                            elif row_list[2].find("⾜球 /大小盘 大")!=-1 and row_list[6].find("赢")!=-1:
                                dicts["大赢"] = dicts["大赢"]+1
                            elif row_list[2].find("⾜球 /大小盘 大")!=-1 and row_list[6].find("输")!=-1:
                                dicts["大输"] = dicts["大输"] + 1
                            elif row_list[2].find("⾜球 /上半场大小盘 大")!=-1 and row_list[6].find("输")!=-1:
                                dicts["大输"] = dicts["大输"] + 1
                            elif row_list[2].find("⾜球 /上半场大小盘 大")!=-1 and row_list[6].find("赢")!=-1:
                                dicts["大赢"] = dicts["大赢"] + 1

                con += 1
                print('---------------分割线,第%s页---------------' % con)
        except Exception as e:
            print('报错：', e)
        finally:
            wb.save('\\'.join(self.pdf_path.split('\\')[:-1]) + '\pdf_excel.xlsx')
            print('写入完成！')
            self.close_pdf()

    # 关闭文件
    def close_pdf(self):
        self.pdf_info.close()


if __name__ == "__main__":

    file_path = 'hxjxx3.pdf'
    pdf_info = PDF(file_path)
    # pdf_info.get_pdf() # 打印pdf基础信息
    # 提取pdf表格数据并保存到excel中,文件保存到跟pdf同一文件路径下
    pdf_info.get_table()
    print(dicts)
