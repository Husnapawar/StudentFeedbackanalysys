"""
Module for user authentication and management.
"""

import os
import json
import hashlib
import secrets
import pandas as pd
from datetime import datetime, timedelta

class UserAuth:
    """
    Class for user authentication and management.
    """
    
    def __init__(self, users_file='data/users.json', sessions_file='data/sessions.json'):
        """
        Initialize the user authentication system.
        
        Parameters:
        -----------
        users_file : str
            Path to the users data file
        sessions_file : str
            Path to the sessions data file
        """
        self.users_file = users_file
        self.sessions_file = sessions_file
        self.ensure_files_exist()
    
    def ensure_files_exist(self):
        """Ensure the user and session files exist."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.sessions_file), exist_ok=True)
        
        # Create users file if it doesn't exist
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump([], f)
        
        # Create sessions file if it doesn't exist
        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w') as f:
                json.dump([], f)
    
    def _hash_password(self, password, salt=None):
        """
        Hash a password with a salt.
        
        Parameters:
        -----------
        password : str
            Password to hash
        salt : str, optional
            Salt to use (if None, a new salt is generated)
            
        Returns:
        --------
        tuple
            (hashed_password, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Hash the password with the salt
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        return hashed, salt
    
    def register_user(self, username, password, email, role='student', full_name=None):
        """
        Register a new user.
        
        Parameters:
        -----------
        username : str
            Username
        password : str
            Password
        email : str
            Email address
        role : str
            User role ('admin', 'instructor', or 'student')
        full_name : str, optional
            Full name of the user
            
        Returns:
        --------
        dict
            Status of the registration
        """
        try:
            # Load existing users
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            # Check if username or email already exists
            for user in users:
                if user['username'] == username:
                    return {
                        'status': 'error',
                        'message': 'Username already exists'
                    }
                if user['email'] == email:
                    return {
                        'status': 'error',
                        'message': 'Email already exists'
                    }
            
            # Hash the password
            hashed_password, salt = self._hash_password(password)
            
            # Create new user
            new_user = {
                'username': username,
                'password_hash': hashed_password,
                'salt': salt,
                'email': email,
                'role': role,
                'full_name': full_name,
                'created_at': datetime.now().isoformat(),
                'last_login': None
            }
            
            # Add to users list
            users.append(new_user)
            
            # Save updated users
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
            
            return {
                'status': 'success',
                'message': 'User registered successfully'
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error registering user: {str(e)}'
            }
    
    def login(self, username, password):
        """
        Login a user.
        
        Parameters:
        -----------
        username : str
            Username
        password : str
            Password
            
        Returns:
        --------
        dict
            Status of the login and session token if successful
        """
        try:
            # Load existing users
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            # Find the user
            user = None
            for u in users:
                if u['username'] == username:
                    user = u
                    break
            
            if user is None:
                return {
                    'status': 'error',
                    'message': 'Invalid username or password'
                }
            
            # Verify password
            hashed_password, _ = self._hash_password(password, user['salt'])
            if hashed_password != user['password_hash']:
                return {
                    'status': 'error',
                    'message': 'Invalid username or password'
                }
            
            # Generate session token
            session_token = secrets.token_hex(32)
            
            # Load existing sessions
            with open(self.sessions_file, 'r') as f:
                sessions = json.load(f)
            
            # Create new session
            new_session = {
                'token': session_token,
                'username': username,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=1)).isoformat()
            }
            
            # Add to sessions list
            sessions.append(new_session)
            
            # Save updated sessions
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)
            
            # Update last login
            for u in users:
                if u['username'] == username:
                    u['last_login'] = datetime.now().isoformat()
            
            # Save updated users
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
            
            return {
                'status': 'success',
                'message': 'Login successful',
                'token': session_token,
                'username': username,
                'role': user['role'],
                'full_name': user['full_name']
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error logging in: {str(e)}'
            }
    
    def logout(self, token):
        """
        Logout a user by invalidating their session token.
        
        Parameters:
        -----------
        token : str
            Session token
            
        Returns:
        --------
        dict
            Status of the logout
        """
        try:
            # Load existing sessions
            with open(self.sessions_file, 'r') as f:
                sessions = json.load(f)
            
            # Remove the session
            sessions = [s for s in sessions if s['token'] != token]
            
            # Save updated sessions
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)
            
            return {
                'status': 'success',
                'message': 'Logout successful'
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error logging out: {str(e)}'
            }
    
    def validate_session(self, token):
        """
        Validate a session token.
        
        Parameters:
        -----------
        token : str
            Session token
            
        Returns:
        --------
        dict
            Status of the validation and user info if successful
        """
        try:
            # Load existing sessions
            with open(self.sessions_file, 'r') as f:
                sessions = json.load(f)
            
            # Find the session
            session = None
            for s in sessions:
                if s['token'] == token:
                    session = s
                    break
            
            if session is None:
                return {
                    'status': 'error',
                    'message': 'Invalid session token'
                }
            
            # Check if session is expired
            expires_at = datetime.fromisoformat(session['expires_at'])
            if expires_at < datetime.now():
                # Remove expired session
                sessions = [s for s in sessions if s['token'] != token]
                with open(self.sessions_file, 'w') as f:
                    json.dump(sessions, f, indent=2)
                
                return {
                    'status': 'error',
                    'message': 'Session expired'
                }
            
            # Load user info
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            # Find the user
            user = None
            for u in users:
                if u['username'] == session['username']:
                    user = u
                    break
            
            if user is None:
                return {
                    'status': 'error',
                    'message': 'User not found'
                }
            
            return {
                'status': 'success',
                'message': 'Session valid',
                'username': user['username'],
                'role': user['role'],
                'full_name': user['full_name'],
                'email': user['email']
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error validating session: {str(e)}'
            }
    
    def get_user_info(self, username):
        """
        Get user information.
        
        Parameters:
        -----------
        username : str
            Username
            
        Returns:
        --------
        dict
            User information
        """
        try:
            # Load existing users
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            # Find the user
            user = None
            for u in users:
                if u['username'] == username:
                    user = u
                    break
            
            if user is None:
                return {
                    'status': 'error',
                    'message': 'User not found'
                }
            
            # Return user info (excluding sensitive data)
            return {
                'status': 'success',
                'username': user['username'],
                'email': user['email'],
                'role': user['role'],
                'full_name': user['full_name'],
                'created_at': user['created_at'],
                'last_login': user['last_login']
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error getting user info: {str(e)}'
            }
    
    def update_user(self, username, updates, current_password=None):
        """
        Update user information.
        
        Parameters:
        -----------
        username : str
            Username
        updates : dict
            Dictionary containing fields to update
        current_password : str, optional
            Current password (required for password changes)
            
        Returns:
        --------
        dict
            Status of the update
        """
        try:
            # Load existing users
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            # Find the user
            user_index = None
            for i, u in enumerate(users):
                if u['username'] == username:
                    user_index = i
                    break
            
            if user_index is None:
                return {
                    'status': 'error',
                    'message': 'User not found'
                }
            
            # Handle password change
            if 'password' in updates:
                # Verify current password
                if current_password is None:
                    return {
                        'status': 'error',
                        'message': 'Current password required for password change'
                    }
                
                hashed_current, _ = self._hash_password(current_password, users[user_index]['salt'])
                if hashed_current != users[user_index]['password_hash']:
                    return {
                        'status': 'error',
                        'message': 'Current password is incorrect'
                    }
                
                # Hash the new password
                hashed_new, salt = self._hash_password(updates['password'])
                users[user_index]['password_hash'] = hashed_new
                users[user_index]['salt'] = salt
                
                # Remove password from updates
                del updates['password']
            
            # Update other fields
            for key, value in updates.items():
                if key in ['email', 'role', 'full_name']:
                    users[user_index][key] = value
            
            # Save updated users
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
            
            return {
                'status': 'success',
                'message': 'User updated successfully'
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error updating user: {str(e)}'
            }
    
    def delete_user(self, username, password):
        """
        Delete a user.
        
        Parameters:
        -----------
        username : str
            Username
        password : str
            Password for verification
            
        Returns:
        --------
        dict
            Status of the deletion
        """
        try:
            # Load existing users
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            # Find the user
            user = None
            for u in users:
                if u['username'] == username:
                    user = u
                    break
            
            if user is None:
                return {
                    'status': 'error',
                    'message': 'User not found'
                }
            
            # Verify password
            hashed_password, _ = self._hash_password(password, user['salt'])
            if hashed_password != user['password_hash']:
                return {
                    'status': 'error',
                    'message': 'Invalid password'
                }
            
            # Remove the user
            users = [u for u in users if u['username'] != username]
            
            # Save updated users
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
            
            # Load existing sessions
            with open(self.sessions_file, 'r') as f:
                sessions = json.load(f)
            
            # Remove user's sessions
            sessions = [s for s in sessions if s['username'] != username]
            
            # Save updated sessions
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)
            
            return {
                'status': 'success',
                'message': 'User deleted successfully'
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error deleting user: {str(e)}'
            }
    
    def list_users(self, role=None):
        """
        List all users, optionally filtered by role.
        
        Parameters:
        -----------
        role : str, optional
            Role to filter by
            
        Returns:
        --------
        list
            List of users (excluding sensitive data)
        """
        try:
            # Load existing users
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            # Filter by role if specified
            if role is not None:
                users = [u for u in users if u['role'] == role]
            
            # Remove sensitive data
            user_list = []
            for user in users:
                user_list.append({
                    'username': user['username'],
                    'email': user['email'],
                    'role': user['role'],
                    'full_name': user['full_name'],
                    'created_at': user['created_at'],
                    'last_login': user['last_login']
                })
            
            return user_list
        
        except Exception as e:
            print(f"Error listing users: {str(e)}")
            return []
    
    def create_admin_if_none(self):
        """
        Create an admin user if none exists.
        
        Returns:
        --------
        dict
            Status of the operation
        """
        try:
            # Load existing users
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            # Check if any admin exists
            admin_exists = any(u['role'] == 'admin' for u in users)
            
            if not admin_exists:
                # Create default admin
                result = self.register_user(
                    username='admin',
                    password='admin123',  # This should be changed immediately
                    email='admin@example.com',
                    role='admin',
                    full_name='System Administrator'
                )
                
                if result['status'] == 'success':
                    return {
                        'status': 'success',
                        'message': 'Default admin user created. Please change the password immediately.',
                        'username': 'admin',
                        'password': 'admin123'
                    }
                else:
                    return result
            
            return {
                'status': 'info',
                'message': 'Admin user already exists'
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error creating admin user: {str(e)}'
            }
