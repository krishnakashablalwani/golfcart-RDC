def sendmail(rollno, name, img):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    sender_email = "krishnakashab@gmail.com"
    receiver_email = "krishnakashab@gmail.com"
    subject = f"Face Recognition Alert: {name} ({rollno})"
    body = f"Alert: {name} with Roll Number {rollno} has been recognized by the system."
    password = "rvzz vuep phqn wyiq"
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(img)
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename="recognized_face.jpg"')
    msg.attach(part)
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print(f"✓ Email sent to {receiver_email}")
    except Exception as e:
        print(f"✗ Failed to send email: {e}")