import streamlit as st
import base64
import cv2
import sqlite3

# ================ Background and Header ===
st.markdown(f'<h1 style="color:#FFFFFF;text-align: center;font-size:36px;">{"A Federated Learning Framework for Breast Cancer Histopathological Image Classification"}</h1>', unsafe_allow_html=True)

# Add static professional background
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0f2fe, #bae6fd, #93c5fd, #60a5fa);

    background-attachment: fixed;
}

/* Style text elements */
h1 {
    color: white !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    font-weight: 700;
    margin-bottom: 2rem;
}

/* Create card-like container for login */
div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stTextInput"]) {
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    padding: 2.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    max-width: 500px;
    margin: 0 auto;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Style input fields */
.stTextInput > div > div > input {
    background-color: rgba(255,255,255,0.9);
    color: #1a365d;
    border-radius: 8px;
    border: 1px solid #cbd5e0;
    padding: 12px;
    font-size: 16px;
    transition: all 0.2s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #4299e1;
    box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.2);
}

/* Style buttons */
.stButton > button {
    background-color: #3182ce;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 12px 24px;
    border: none;
    width: 100%;
    margin-top: 1rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.stButton > button:hover {
    background-color: #2c5282;
    transform: translateY(-1px);
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
}

/* Style columns */
div[data-testid="column"] {
    padding: 0 0.5rem;
}

/* Style success and error messages */
div[data-testid="stException"] {
    padding: 1rem;
    border-radius: 8px;
    margin-top: 1rem;
}

/* Style login header */
div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stTextInput"])::before {
    content: "Login Here";
    color: #1a365d;
    font-size: 28px;
    font-weight: 600;
    text-align: center;
    display: block;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ----------------------

# Function to create a database connection
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn

# Function to create a new user
def create_user(conn, user):
    sql = ''' INSERT INTO users(name, password, email, phone)
              VALUES(?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, user)
    conn.commit()
    return cur.lastrowid

# Function to validate user credentials
def validate_user(conn, name, password):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE name=? AND password=?", (name, password))
    user = cur.fetchone()
    if user:
        return True, user[1]  # Return True and user name
    return False, None

# Main function
def main():
    # Create a database connection
    conn = create_connection("dbs.db")
    if conn is not None:
        # Create users table if it doesn't exist
        conn.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY,
                     name TEXT NOT NULL,
                     password TEXT NOT NULL,
                     email TEXT NOT NULL UNIQUE,
                     phone TEXT NOT NULL);''')
        
        # Login form container
        with st.container():
            name = st.text_input("User name")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.button("Login")
                
            if login_button:
                is_valid, user_name = validate_user(conn, name, password)
                if is_valid:
                    st.success(f"Welcome back, {user_name}! Login successful!")
                    
                    import subprocess
                    subprocess.run(['python','-m','streamlit','run','app.py'])
                else:
                    st.error("Invalid user name or password!")
                    
        # Close the database connection
        conn.close()
    else:
        st.error("Error! cannot create the database connection.")

if __name__ == '__main__':
    main()