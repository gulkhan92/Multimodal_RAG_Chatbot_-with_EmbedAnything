from rag_multimodal.database import SessionLocal, engine, Base, User, Role
from rag_multimodal.auth_utils import get_password_hash
from rag_multimodal.settings import Settings

def seed_database():
    # Create tables
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    try:
        # Check if roles exist
        admin_role = db.query(Role).filter(Role.name == "admin").first()
        if not admin_role:
            admin_role = Role(name="admin")
            db.add(admin_role)

        viewer_role = db.query(Role).filter(Role.name == "viewer").first()
        if not viewer_role:
            viewer_role = Role(name="viewer")
            db.add(viewer_role)

        # Check if admin user exists
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            settings = Settings.from_env()
            hashed_password = get_password_hash(settings.default_admin_password)
            admin_user = User(username="admin", email="admin@example.com", hashed_password=hashed_password)
            admin_user.roles.append(admin_role)
            admin_user.roles.append(viewer_role)
            db.add(admin_user)

        db.commit()
        print("Database seeded successfully.")
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()