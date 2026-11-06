from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

import os
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server thread safety
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__, static_folder='Static')
app.secret_key = 'secret_123'

# Database configuration with graceful fallback to SQLite if MySQL is unreachable
default_mysql = 'mysql+pymysql://root:root@localhost/ml_app'
configured_db = os.environ.get('DATABASE_URL', default_mysql)

try:
    from sqlalchemy import create_engine
    test_engine = create_engine(configured_db, connect_args={'connect_timeout': 1})
    with test_engine.connect():
        pass
    app.config['SQLALCHEMY_DATABASE_URI'] = configured_db
except Exception:
    # Fallback to local SQLite database for standalone environments
    db_file = os.path.join(os.path.abspath(os.getcwd()), 'ml_app.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_file}'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy & Bcrypt
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# User Database Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password

# Upload Configuration
UPLOAD_FOLDER = os.path.join(os.path.abspath(os.getcwd()), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global dataset placeholder with automatic workspace preloading
uploaded_data = None
default_csv = os.path.join(UPLOAD_FOLDER, 'diabetes.csv')
if not os.path.exists(default_csv):
    default_csv = os.path.join(os.path.abspath(os.getcwd()), 'diabetes.csv')

if os.path.exists(default_csv):
    try:
        uploaded_data = pd.read_csv(default_csv)
    except Exception:
        uploaded_data = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception:
            db.session.rollback()
            flash('Error: Username or email already registered.', 'danger')
            
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password']
        
        try:
            user = User.query.filter_by(email=email).first()
            if user and bcrypt.check_password_hash(user.password, password):
                session['user_id'] = user.id
                session['username'] = user.username
                flash(f'Welcome back, {user.username}!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid email or password. Please try again.', 'danger')
        except Exception as e:
            flash(f'Database error: {e}', 'danger')
            
    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    username = session.get('username', 'Clinician')
    return render_template('dashboard.html', username=username)


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global uploaded_data
    file_uploaded = False
    columns, data = [], []
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part provided in request.', 'danger')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'warning')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_uploaded = True
            
            try:
                uploaded_data = pd.read_csv(file_path)
                columns = uploaded_data.columns.tolist()
                data = uploaded_data.head(10).values.tolist()
                flash('File uploaded and parsed successfully!', 'success')
                return render_template('upload.html',
                                       file_uploaded=file_uploaded,
                                       columns=columns,
                                       data=data)
            except Exception as e:
                flash(f'Error reading CSV file: {e}', 'danger')
                return redirect(request.url)
                
        flash('Invalid file format. Please upload a standard CSV file.', 'danger')
        return redirect(request.url)
        
    return render_template('upload.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    global uploaded_data
    if uploaded_data is None:
        flash('No dataset uploaded. Please upload a dataset first.', 'warning')
        return redirect(url_for('upload'))
        
    try:
        # All columns except the last are features
        X = uploaded_data.iloc[:, :-1]
        # Last column is target outcome
        y = uploaded_data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        models_dir = os.path.join(os.path.abspath(os.getcwd()), 'models')
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(rf_model, os.path.join(models_dir, "model.pkl"))
        
        flash('Training completed successfully! Model serialized and ready for inference.', 'success')
        return render_template('train.html')
    except Exception as e:
        flash(f"Error during model training: {e}", 'danger')
        return redirect(url_for('upload'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global uploaded_data
    if request.method == 'POST':
        try:
            inputs = {
                "Pregnancies": int(request.form['Pregnancies']),
                "Glucose Level": float(request.form['Glucose Level']),
                "Blood Pressure": float(request.form['Blood Pressure']),
                "Skin Thickness": float(request.form['Skin Thickness']),
                "Insulin Level": float(request.form['Insulin Level']),
                "BMI": float(request.form['BMI']),
                "Diabetes Pedigree Function": float(request.form['Diabetes Pedigree Function']),
                "Age": int(request.form['Age'])
            }
            
            input_values = list(inputs.values())
            model_path = os.path.join(os.path.abspath(os.getcwd()), 'models', 'model.pkl')
            
            # If model file does not exist, auto-train if dataset is available
            if not os.path.exists(model_path):
                if uploaded_data is not None:
                    train()
                else:
                    flash('No trained model found. Please train a model first.', 'warning')
                    return redirect(url_for('upload'))
            
            model = joblib.load(model_path)
            prediction = model.predict([input_values])[0]
            result = "Diabetes Detected" if prediction == 1 else "No Diabetes Detected"
            
            return render_template('result.html', inputs=inputs, result=result)
        except Exception as e:
            flash(f"Error during prediction: {e}", 'danger')
            return redirect(url_for('predict'))
            
    return render_template('predict.html')


@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    global uploaded_data
    if uploaded_data is None:
        flash('No dataset uploaded. Please upload a dataset first.', 'warning')
        return redirect(url_for('upload'))
        
    scatter_plot = generate_scatter_plot(uploaded_data)
    return render_template('visualize.html', scatter_plot=scatter_plot)


def generate_scatter_plot(data):
    try:
        plot_df = data.copy()
        
        # Ensure Pregnancies is numeric
        if 'Pregnancies' in plot_df.columns:
            plot_df['Pregnancies'] = pd.to_numeric(plot_df['Pregnancies'], errors='coerce').fillna(0).astype(int)
        
        if 'Pregnancies' in plot_df.columns and 'Glucose' in plot_df.columns:
            plt.figure(figsize=(9, 5.5), facecolor='#08090d')
            ax = plt.axes()
            ax.set_facecolor('#10121a')
            
            hue_col = 'Outcome' if 'Outcome' in plot_df.columns else None
            palette = {0: '#10b981', 1: '#ff4d6d'} if hue_col else None
            
            sns.scatterplot(
                data=plot_df,
                x='Pregnancies',
                y='Glucose',
                hue=hue_col,
                palette=palette,
                alpha=0.85,
                s=70,
                edgecolor='#ffffff',
                linewidth=0.6,
                ax=ax
            )
            
            plt.title("Clinical Correlation: Pregnancies vs. Glucose Level", color='#fbfbfd', fontsize=13, fontweight='bold', pad=15)
            plt.xlabel("Number of Pregnancies", color='#d1d5db', fontsize=11, labelpad=10)
            plt.ylabel("Glucose Level (mg/dL)", color='#d1d5db', fontsize=11, labelpad=10)
            
            ax.tick_params(colors='#9ca3af', labelsize=10)
            for spine in ax.spines.values():
                spine.set_color('#26283b')
            ax.grid(True, linestyle='--', alpha=0.3, color='#26283b')
            
            if hue_col:
                legend = plt.legend(title='Clinical Risk', labels=['Negative (0)', 'Positive (1)'], facecolor='#10121a', edgecolor='#26283b')
                for text in legend.get_texts():
                    text.set_color('#e2e8f0')
                legend.get_title().set_color('#9ca3af')
                
            plt.tight_layout()
            
            img_dir = os.path.join(app.root_path, 'Static', 'images')
            os.makedirs(img_dir, exist_ok=True)
            filepath = os.path.join(img_dir, 'scatter_plot.png')
            plt.savefig(filepath, dpi=160, facecolor='#08090d', edgecolor='none')
            plt.close()
            return filepath
    except Exception as e:
        print(f"Error generating scatter plot: {e}")
        return None


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
    app.run(debug=False, host='0.0.0.0', port=5000)
