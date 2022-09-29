from crypt import methods
from genericpath import exists
import shutil
from urllib import request
from app import app 
from flask import render_template,session, redirect, url_for,request
import pymysql
import os
import pathlib

connection = pymysql.connect(host = 'localhost', user = 'root', password = '', database = 'guardians')
cursor = connection.cursor()
app.secret_key = 'mysecretkey'

app.config['SIGNATURE_UPLOADS'] = '/Users/alexandroalvin/Documents/Guardians_FrontEnd/app/static/img/signatures'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])


@app.route('/admin/dashboard',methods=['GET'])
def admin_dashboard():
    if "username" in session:
        username = session['username']
        signature_registered = cursor.execute('SELECT signature_registered FROM user WHERE username = %s', username)
        signature_registered = cursor.fetchone()

        signature_comparison = cursor.execute('SELECT signature_comparison FROM user WHERE username = %s', username)
        signature_comparison = cursor.fetchone()

        document_verified = cursor.execute('SELECT document_verified FROM user WHERE username = %s', username)
        document_verified = cursor.fetchone()

        dataKaryawan = cursor.execute('SELECT * FROM karyawan')
        dataKaryawan = cursor.fetchall()

        userID = cursor.execute('SELECT userID FROM user WHERE username = %s', username)
        userID = cursor.fetchone()
        history = cursor.execute('SELECT * FROM history WHERE userID = %s', userID)
        history = cursor.fetchall()

        return render_template('admin/dashboard.html', sgn = signature_registered[0], sgn_cmp = signature_comparison[0], doc = document_verified[0], usernames = username, history = history,len = len(history), dataKaryawan = dataKaryawan, lenKaryawan = len(dataKaryawan))
    else:
        return redirect(url_for('login'))
        
@app.route('/admin/compare')
def admin_compare():
    if "username" in session:
        username = session['username']
        karyawan = cursor.execute('SELECT namaKaryawan FROM karyawan')
        karyawan = cursor.fetchall()
        return render_template('admin/compare.html', usernames = username, karyawan = karyawan, len = len(karyawan))
    else:
        return redirect(url_for('login'))

@app.route('/admin/add')
def admin_add():
    if "username" in session:
        username = session['username']
        return render_template('admin/add.html', usernames = username)
    else:
        return redirect(url_for('login'))

@app.route('/admin/addUsers', methods=['GET','POST'])
def admin_add_user():
    if "username" in session:
        username = session['username']
        if request.method == 'POST':
            nameKaryawan = request.form['namaKaryawan']
            basedir = os.path.abspath(os.path.dirname(__file__))
            file = request.files['signature']
            file1 = request.files['signature1']
            file2 = request.files['signature2']
            file3 = request.files['signature3']
            file4 = request.files['signature4']

            directories = pathlib.Path(app.config['SIGNATURE_UPLOADS'], nameKaryawan).mkdir(exist_ok=True)

            # save the name of directory
            directory = os.path.join(app.config['SIGNATURE_UPLOADS'], nameKaryawan)
            print(directory)
            filename = 'File_000'+ '.png'
            filename1 = 'File_000(1)'+ '.png'
            filename2 = 'File_000(2)'+ '.png'
            filename3 = 'File_000(3)'+ '.png'
            filename4 = 'File_000(4)'+ '.png'
            file.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename))
            file1.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename1))
            file2.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename2))
            file3.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename3))
            file4.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename4))

            cursor.execute('INSERT INTO karyawan (namaKaryawan, pathfile) VALUES (%s, %s)', (nameKaryawan, directory))
            connection.commit()

            rgst = cursor.execute('SELECT signature_registered FROM user WHERE username = %s', username)
            rgst = cursor.fetchone()

            rgst = rgst[0] + 1
            cursor.execute('UPDATE user SET signature_registered = %s', (rgst))
            connection.commit()
            return redirect(url_for('admin_dashboard'))
    else:
        return redirect(url_for('login'))

@app.route('/admin/editUsers/<string:id>', methods=['GET','POST'])
def admin_edit(id):
    if "username" in session:
        username = session['username']
        print(id)
        if request.method == 'POST':
            if request.files['signature'].filename == '':
                nameKaryawan = request.form['namaKaryawan']
                # get the directory and replace the name of directory with the new name
                directory = cursor.execute('SELECT pathfile FROM karyawan WHERE karyawanID = %s', id)
                directory = cursor.fetchone()
                directory = directory[0]
                directory = directory.replace(directory.split('/')[-1], nameKaryawan)
                print(directory)
                # replace the name of folder in my directory

                olddirectories = cursor.execute('SELECT pathfile FROM karyawan WHERE karyawanID = %s', id)
                olddirectories = cursor.fetchone()
                olddirectories = olddirectories[0]
                os.rename(olddirectories, directory)
                # update the name of directory
                cursor.execute('UPDATE karyawan SET namaKaryawan = %s, pathfile = %s WHERE karyawanID = %s', (nameKaryawan, directory, id))
                connection.commit()
                return redirect(url_for('admin_dashboard'))
            # elif update 1 file and the others not update, but the file where not update, the name of file will be the same
            elif request.files['signature'].filename != '' and request.files['signature1'].filename == '' and request.files['signature2'].filename == '' and request.files['signature3'].filename == '' and request.files['signature4'].filename == '':
                nameKaryawan = request.form['namaKaryawan']
                basedir = os.path.abspath(os.path.dirname(__file__))
                file = request.files['signature']
                # get the directory and replace the name of directory with the new name
                directory = cursor.execute('SELECT pathfile FROM karyawan WHERE karyawanID = %s', id)
                directory = cursor.fetchone()
                directory = directory[0]
                directory = directory.replace(directory.split('/')[-1], nameKaryawan)
                print(directory)
                # replace the name of folder in my directory

                olddirectories = cursor.execute('SELECT pathfile FROM karyawan WHERE karyawanID = %s', id)
                olddirectories = cursor.fetchone()
                olddirectories = olddirectories[0]
                os.rename(olddirectories, directory)
                # update the name of directory
                cursor.execute('UPDATE karyawan SET namaKaryawan = %s, pathfile = %s WHERE karyawanID = %s', (nameKaryawan, directory, id))
                connection.commit()
                # save the name of directory
                filename = 'File_000'+ '.png'
                file.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename))
                return redirect(url_for('admin_dashboard'))
            elif request.files['signature'].filename != '' and request.files['signature1'].filename != '' and request.files['signature2'].filename == '' and request.files['signature3'].filename == '' and request.files['signature4'].filename == '':
                nameKaryawan = request.form['namaKaryawan']
                basedir = os.path.abspath(os.path.dirname(__file__))
                file = request.files['signature']
                file1 = request.files['signature1']
                # get the directory and replace the name of directory with the new name
                directory = cursor.execute('SELECT pathfile FROM karyawan WHERE karyawanID = %s', id)
                directory = cursor.fetchone()
                directory = directory[0]
                directory = directory.replace(directory.split('/')[-1], nameKaryawan)
                print(directory)
                # replace the name of folder in my directory

                olddirectories = cursor.execute('SELECT pathfile FROM karyawan WHERE karyawanID = %s', id)
                olddirectories = cursor.fetchone()
                olddirectories = olddirectories[0]
                os.rename(olddirectories, directory)
                # update the name of directory
                cursor.execute('UPDATE karyawan SET namaKaryawan = %s, pathfile = %s WHERE karyawanID = %s', (nameKaryawan, directory, id))
                connection.commit()
                # save the name of directory
                filename = 'File_000'+ '.png'
                filename1 = 'File_000(1)'+ '.png'
                file.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename))
                file1.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename1))
                return redirect(url_for('admin_dashboard'))
            elif request.files['signature'].filename != '' and request.files['signature1'].filename != '' and request.files['signature2'].filename != '' and request.files['signature3'].filename == '' and request.files['signature4'].filename == '':
                nameKaryawan = request.form['namaKaryawan']
                basedir = os.path.abspath(os.path.dirname(__file__))
                file = request.files['signature']
                file1 = request.files['signature1']
                file2 = request.files['signature2']
                # get the directory and replace the name of directory with the new name
                directory = cursor.execute('SELECT pathfile FROM karyawan WHERE karyawanID = %s', id)
                directory = cursor.fetchone()
                directory = directory[0]
                directory = directory.replace(directory.split('/')[-1], nameKaryawan)
                print(directory)
                # replace the name of folder in my directory

                olddirectories = cursor.execute('SELECT pathfile FROM karyawan WHERE karyawanID = %s', id)
                olddirectories = cursor.fetchone()
                olddirectories = olddirectories[0]
                os.rename(olddirectories, directory)
                # update the name of directory
                cursor.execute('UPDATE karyawan SET namaKaryawan = %s, pathfile = %s WHERE karyawanID = %s', (nameKaryawan, directory, id))
                connection.commit()
                # save the name of directory
                filename = 'File_000'+ '.png'
                filename1 = 'File_000(1)'+ '.png'
                filename2 = 'File_000(2)'+ '.png'
                file.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename))
                file1.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename1))
                file2.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename2))
                return redirect(url_for('admin_dashboard'))
            elif request.files['signature'].filename != '' and request.files['signature1'].filename != '' and request.files['signature2'].filename != '' and request.files['signature3'].filename != '' and request.files['signature4'].filename == '':
                nameKaryawan = request.form['namaKaryawan']
                basedir = os.path.abspath(os.path.dirname(__file__))
                file = request.files['signature']
                file1 = request.files['signature1']
                file2 = request.files['signature2']
                file3 = request.files['signature3']
                # get the directory and replace the name of directory with the new name
                directory = cursor.execute('SELECT pathfile FROM karyawan WHERE karyawanID = %s', id)
                directory = cursor.fetchone()
                directory = directory[0]
                directory = directory.replace(directory.split('/')[-1], nameKaryawan)
                print(directory)
                # replace the name of folder in my directory

                olddirectories = cursor.execute('SELECT pathfile FROM karyawan WHERE karyawanID = %s', id)
                olddirectories = cursor.fetchone()
                olddirectories = olddirectories[0]
                os.rename(olddirectories, directory)
                # update the name of directory
                cursor.execute('UPDATE karyawan SET namaKaryawan = %s, pathfile = %s WHERE karyawanID = %s', (nameKaryawan, directory, id))
                connection.commit()
                # save the name of directory
                filename = 'File_000'+ '.png'
                filename1 = 'File_000(1)'+ '.png'
                filename2 = 'File_000(2)'+ '.png'
                filename3 = 'File_000(3)'+ '.png'
                file.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename))
                file1.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename1))
                file2.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename2))
                file3.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename3))
                return redirect(url_for('admin_dashboard'))
            else:
                basedir = os.path.abspath(os.path.dirname(__file__))
                oldfilename = cursor.execute('SELECT pathfile FROM karyawan WHERE karyawanID = %s', id)
                oldfilename = cursor.fetchone()
                olddirectories = os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],oldfilename[0])
                print(olddirectories)
                if os.path.exists(olddirectories):
                    os.remove(olddirectories)   

                nameKaryawan = request.form['namaKaryawan']
                basedir = os.path.abspath(os.path.dirname(__file__))
                file = request.files['signature']
                file1 = request.files['signature1']
                file2 = request.files['signature2']
                file3 = request.files['signature3']
                file4 = request.files['signature4']

                directories = pathlib.Path(app.config['SIGNATURE_UPLOADS'], nameKaryawan).mkdir(exist_ok=True)

                # save the name of directory
                directory = os.path.join(app.config['SIGNATURE_UPLOADS'], nameKaryawan)
                print(directory)
                filename = 'File_000'+ '.png'
                filename1 = 'File_000(1)'+ '.png'
                filename2 = 'File_000(2)'+ '.png'
                filename3 = 'File_000(3)'+ '.png'
                filename4 = 'File_000(4)'+ '.png'
                file.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename))
                file1.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename1))
                file2.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename2))
                file3.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename3))
                file4.save(os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],nameKaryawan,filename4))
                cursor.execute('UPDATE karyawan SET namaKaryawan = %s, pathfile = %s WHERE karyawanID = %s', (nameKaryawan, directories, id))
                connection.commit()
                return redirect(url_for('admin_dashboard'))
    else:
        return redirect(url_for('login'))



@app.route('/admin/delete/<string:id_data>', methods=['GET','POST'])
def admin_delete(id_data):
    if "username" in session:
        username = session['username']
        directory = cursor.execute('SELECT pathfile FROM karyawan WHERE karyawanID = %s', id_data)
        directory = cursor.fetchone()

        basedir = os.path.abspath(os.path.dirname(__file__))
        directory = os.path.join(basedir, app.config['SIGNATURE_UPLOADS'],directory[0])
        print(directory)
        if os.path.exists(directory):
            shutil.rmtree(directory)
        
        cursor.execute('DELETE FROM karyawan WHERE karyawanID = %s', id_data)
        connection.commit()
        print(username)

        rgst = cursor.execute('SELECT signature_registered FROM user where username = %s', username)
        rgst = cursor.fetchone()
        print(rgst)
        rgst = rgst[0]
        print(rgst)
        rgst = rgst - 1

        cursor.execute('UPDATE user SET signature_registered = %s', (rgst))
        connection.commit()
        return redirect(url_for('admin_dashboard'))
    else:
        return redirect(url_for('login'))
