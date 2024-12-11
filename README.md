# Cloud-Computing  
Backend API for the TRASCAN application  

# Members
Allgiant O'Neill Feltony Asywal - C133B4KY0404    
Antonieta Aryuka Paskalia Nggotu - C269B4KX0591  

# Cloud Architecture
![Copy of cloud-architecture-trascan drawio](https://github.com/user-attachments/assets/3266cfde-6513-4309-ac99-0dc848559732)

# API ENDPOINT
[TRASCAN API Documentation](https://lacy-sting-8e9.notion.site/TRASCAN-API-15328cfc24d580c8a8a2f44c637b7e00)

## **Running the API Locally**

## *Classification and Articles API*

- Clone the repository to your local machine and change the directory to trascan-be
```bash
git clone https://github.com/TRASCAN-app/Cloud-Computing.git
cd Cloud-Computing/trascan-be/
```

- Install All Dependencies

```bash
pip install -r requirements.txt
```

- Set Up Service Account Credentials  
  Upload the credentials.json file (downloaded from your Google Cloud Service Account) into the project directory.

- Set Up the Project on GCP  
  Ensure that your project is linked to a billing account so that resources can be used without restrictions (e.g., Cloud Run, Cloud Storage, Firestore).

- Run the server

```bash
uvicorn main:app --reload
```
## *Locations API*
- Clone the repository to your local machine and change the directory to maps-demo
  
```bash
git clone https://github.com/TRASCAN-app/Cloud-Computing.git
cd Cloud-Computing/endpoint-map-main/
```

- Install All Dependencies

```bash
npm install
```

- Create a file named .env in the root directory of your project  
  Add your Google Maps API key to the .env file:  
  
  ```bash
   GOOGLE_MAPS_API_KEY=your-api-key-here
  ```

- Set Up Service Account Credentials  
  Upload the credentials.json file (downloaded from your Google Cloud Service Account) into the project directory.

- Set Up the Project on GCP  
  Ensure that your project is linked to a billing account so that resources can be used without restrictions (e.g., Cloud Run, Places API, etc.)

- Run the server

```bash
node index.js
```
