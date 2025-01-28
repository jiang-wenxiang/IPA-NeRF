import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

import sys

import torch
import pynvml  # pip install nvidia-ml-py3


def send_dict(title, dict_msg: dict):
    smtp_server_address_list = ['smtp.qq.com']
    smtp_server_port_list = [465]
    your_email_address_list = ["change-for-yourself-email"]
    your_email_password_list = ["change-for-yourself-email-smtp-password"]
    computer_name = "my computer"

    for i in range(len(smtp_server_address_list)):
        smtp_server_address = smtp_server_address_list[i]
        smtp_server_port = smtp_server_port_list[i]
        your_email_address = your_email_address_list[i]
        your_email_password = your_email_password_list[i]

        if (your_email_address != "change-for-yourself-email" and
                your_email_password != "change-for-yourself-email-smtp-password"):

            # 1. 连接邮箱服务器
            con = smtplib.SMTP_SSL(smtp_server_address, smtp_server_port)
            # 2. 登录邮箱
            con.login(your_email_address, your_email_password)
            # 2. 准备数据
            # 创建邮件对象
            msg = MIMEMultipart()
            # 设置邮件主题
            subject = Header(title, 'utf-8').encode()
            msg['Subject'] = subject
            # 设置邮件发送者
            msg['From'] = your_email_address+' <'+your_email_address+'>'
            # 设置邮件接受者
            msg['To'] = your_email_address
            # 添加⽂文字内容

            text_str = "From [" + computer_name + "] : <br>"
            text_str += "<table>"
            for k in dict_msg:
                text_str += "<tr>"
                text_str += "<td>" + str(k) + "</td><td>" + str(dict_msg[k]) + "</td>"
                text_str += "</tr>"
            text_str += "</table>"

            text = MIMEText(text_str, 'html', 'utf-8')
            msg.attach(text)
            # 3.发送邮件
            con.sendmail(your_email_address, your_email_address, msg.as_string())
            con.quit()
        else:
            print("Please set the correct email address and email smtp password.")


def send_list_for_dict(title, list_dict_msg: list):
    smtp_server_address_list = ['smtp.qq.com']
    smtp_server_port_list = [465]
    your_email_address_list = ["change-for-yourself-email"]
    your_email_password_list = ["change-for-yourself-email-smtp-password"]
    computer_name = "my computer"

    for i in range(len(smtp_server_address_list)):
        smtp_server_address = smtp_server_address_list[i]
        smtp_server_port = smtp_server_port_list[i]
        your_email_address = your_email_address_list[i]
        your_email_password = your_email_password_list[i]

        if (your_email_address != "change-for-yourself-email" and
                your_email_password != "change-for-yourself-email-smtp-password"):

            # 1. 连接邮箱服务器
            con = smtplib.SMTP_SSL(smtp_server_address, smtp_server_port)
            # 2. 登录邮箱
            con.login(your_email_address, your_email_password)
            # 2. 准备数据
            # 创建邮件对象
            msg = MIMEMultipart()
            # 设置邮件主题
            subject = Header(title, 'utf-8').encode()
            msg['Subject'] = subject
            # 设置邮件发送者
            msg['From'] = your_email_address+' <'+your_email_address+'>'
            # 设置邮件接受者
            msg['To'] = your_email_address
            # 添加⽂文字内容

            text_str = "From [" + computer_name + "] : <br>"
            text_str += "<br>"
            for dict_msg in list_dict_msg:
                text_str += "------- " + dict_msg["title"] + " -------<br>"
                text_str += "<table>"
                for k in dict_msg:
                    if k != "title":
                        text_str += "<tr>"
                        text_str += "<td>" + str(k) + "</td><td>" + str(dict_msg[k]) + "</td>"
                        text_str += "</tr>"
                text_str += "</table>"
                text_str += "<br><br>"
            text_str += "<br>"

            text = MIMEText(text_str, 'html', 'utf-8')
            msg.attach(text)
            # 3.发送邮件
            con.sendmail(your_email_address, your_email_address, msg.as_string())
            con.quit()
        else:
            print("Please set the correct email address and email smtp password.")


# pip install nvidia-ml-py3
if __name__ == '__main__':
    gpu_state = {}
    pynvml.nvmlInit()

    min_free_MB = 20000
    gpu_jump = [2, 3]
    email_title = "GPU Free Memory State Report"

    usual_report = False
    TIME_REPORT_INTERVAL = 60 * 60

    while True:
        send_email = False
        send_email_dict = {}

        if torch.cuda.is_available():
            gpu_num = torch.cuda.device_count()

            gpu_state_str = "\rGPU Free Memory State   "
            # update GPU state
            for i in range(gpu_num):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_memory = meminfo.free / (1024 ** 2)
                gpu_state[i] = free_memory
                if i in gpu_jump:
                    gpu_state_str += "[skip] > GPU:" + str(i) + " - " + str(free_memory) + " MB    "
                    send_email_dict["GPU "+str(i) + ": "] = str(free_memory) + " MB [skip]"
                else:
                    gpu_state_str += "GPU:"+str(i)+" - " + str(free_memory) + " MB    "
                    if free_memory > min_free_MB:
                        send_email = True
                        send_email_dict["GPU " + str(i) + ": "] = "<strong>" + str(free_memory) + " MB</strong>"
                    else:
                        send_email_dict["GPU " + str(i) + ": "] = str(free_memory) + " MB"

            print(gpu_state_str, end='')
            sys.stdout.flush()

        if usual_report:
            send_email = True

        if send_email:
            send_dict(email_title, send_email_dict)
            time.sleep(TIME_REPORT_INTERVAL)  # wait TIME_REPORT_INTERVAL
        else:
            time.sleep(1 * 10)  # wait 10 s



