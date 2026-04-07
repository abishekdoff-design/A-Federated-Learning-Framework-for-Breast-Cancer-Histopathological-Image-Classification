# ====================== IMPORTS ======================
import streamlit as st
import sqlite3
import re
import subprocess

# ====================== PAGE STYLING ======================
def set_styles():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        h1, h2 {
            text-align: center;
            color: #2c3e50;
        }

        h1 {
            font-size: 30px;
            font-weight: 700;
        }

        h2 {
            font-size: 24px;
            font-weight: 600;
            color: #3498db;
        }

        .form-container {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            margin: 0 auto;
        }

        .stTextInput > div > div > input {
            background-color: rgba(255,255,255,0.9);
            border-radius: 8px;
            padding: 10px;
            border: 2px solid #ccc;
            font-size: 16px;
        }

        div[data-baseweb="button"] > button {
            background: linear-gradient(90deg, #3498db, #2980b9);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
        }

        div[data-baseweb="button"] > button:hover {
            background: linear-gradient(90deg, #2980b9, #1f5f99);
        }
        </style>
    """, unsafe_allow_html=True)

# ====================== DATABASE FUNCTIONS ======================
def create_connection(db_file):
    try:
        return sqlite3.connect(db_file)
    except sqlite3.Error as e:
        st.error(f"Database connection error: {e}")
        return None

def initialize_db(conn):
    try:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY,
                            name TEXT NOT NULL,
                            password TEXT NOT NULL,
                            email TEXT NOT NULL UNIQUE,
                            phone TEXT NOT NULL
                        );''')
    except sqlite3.Error as e:
        st.error(f"Database initialization error: {e}")

def create_user(conn, user):
    sql = ''' INSERT INTO users(name, password, email, phone)
              VALUES(?,?,?,?) '''
    try:
        cur = conn.cursor()
        cur.execute(sql, user)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        st.error("Email already registered!")
        return False
    except sqlite3.Error as e:
        st.error(f"Error adding user: {e}")
        return False

def user_exists(conn, email):
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        return cur.fetchone() is not None
    except sqlite3.Error as e:
        st.error(f"Error checking user: {e}")
        return False

# ====================== VALIDATIONS ======================
def validate_email(email):
    pattern = r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
    return re.match(pattern, email)

def validate_phone(phone):
    pattern = r'^[6-9]\d{9}$'
    return re.match(pattern, phone)

# ====================== MAIN APP ======================
def main():
    set_styles()

    st.markdown('<h1>Histopathological Image Classification</h1>', unsafe_allow_html=True)
    st.markdown('<h2>Register Here !!!</h2>', unsafe_allow_html=True)

    conn = create_connection("dbs.db")
    if conn is not None:
        initialize_db(conn)

        # ============ Styled Form Container ============
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        with st.form("registration_form"):
            name = st.text_input("Enter your name", max_chars=50)
            password = st.text_input("Enter your password", type="password", max_chars=30)
            confirm_password = st.text_input("Confirm your password", type="password", max_chars=30)
            email = st.text_input("Enter your email", max_chars=50)
            phone = st.text_input("Enter your phone number", max_chars=10)

            col1, col2 = st.columns(2)
            with col1:
                register_button = st.form_submit_button("REGISTER")
            with col2:
                login_button = st.form_submit_button("LOGIN")

        st.markdown('</div>', unsafe_allow_html=True)

        # ============ Form Logic ============
        if register_button:
            if not all([name, password, confirm_password, email, phone]):
                st.error("All fields are required!")
            elif password != confirm_password:
                st.error("Passwords do not match!")
            elif not validate_email(email):
                st.error("Invalid email format!")
            elif not validate_phone(phone):
                st.error("Phone number must start with 6-9 and be 10 digits!")
            elif user_exists(conn, email):
                st.warning("User with this email already exists!")
            else:
                if create_user(conn, (name, password, email, phone)):
                    st.success("🎉 User registered successfully!")

        elif login_button:
            conn.close()
            try:
                subprocess.Popen(["streamlit", "run", "Login.py"])
            except Exception as e:
                st.error(f"Failed to launch login page: {e}")

if __name__ == '__main__':
    main()
