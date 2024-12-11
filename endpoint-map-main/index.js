const express = require('express');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 8080;


app.use(express.json());


const calculateDistance = (lat1, lon1, lat2, lon2) => {
    const toRadians = (degree) => (degree * Math.PI) / 180;
    const R = 6371; 
    const dLat = toRadians(lat2 - lat1);
    const dLon = toRadians(lon2 - lon1);
    const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c; 
};


app.get('/recycling-locations', async (req, res) => {
    const { lat, lon, radius } = req.query;
    const apiKey = 'AIzaSyAaZsCeuDI3KtLpPO0wBleci7Q7e1U3_QY'; 

   
    if (!lat || !lon || !radius || isNaN(lat) || isNaN(lon) || isNaN(radius)) {
        return res.status(400).json({
            error: 'Parameter lat, lon, dan radius diperlukan dan harus berupa angka. Contoh: ?lat=-6.2088&lon=106.8456&radius=1000'
        });
    }

    try {
        
        const url = `https://maps.googleapis.com/maps/api/place/nearbysearch/json`;

        
        const response = await axios.get(url, {
            params: {
                location: `${lat},${lon}`,
                radius: radius,
                keyword: 'recycling',
                key: apiKey
            }
        });

        
        if (response.data.status !== 'OK') {
            return res.status(400).json({
                error: `Google API Error: ${response.data.status}`,
                message: response.data.error_message || 'Periksa parameter API Key atau input Anda.'
            });
        }

       
        const results = response.data.results.map(place => ({
            name: place.name,
            distance_km: calculateDistance(
                parseFloat(lat),
                parseFloat(lon),
                place.geometry.location.lat,
                place.geometry.location.lng
            ),
            address: place.vicinity
        }));

        
        res.json({ locations: results });
    } catch (error) {
        console.error('Error saat memanggil Google API:', error.message);
        res.status(500).json({ error: 'Terjadi kesalahan saat menghubungi Google Places API.' });
    }
});


app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
