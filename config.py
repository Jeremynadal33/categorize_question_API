import os

FB_APP_ID = 779991819235845
SECRET_KEY = ',y"xR=gCltreW`\n\rMil9,xJ}'

basedir = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
