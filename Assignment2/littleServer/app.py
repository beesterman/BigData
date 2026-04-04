from flask import Flask, send_file

app = Flask("__quickServer__")


@app.route('/')
def home():
    return send_file('./spamEmailDataPartial.csv', as_attachment=True)

if __name__ == '__main__':
    app.run()