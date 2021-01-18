import os

SECRET_KEY = ',y"xR=gCltreW`\n\rMil9,xJ}'


if os.environ.get('DATABASE_URL') is None :
    basedir = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
    FB_APP_ID = 779991819235845
else:
    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
    FB_APP_ID = 721286305492889
