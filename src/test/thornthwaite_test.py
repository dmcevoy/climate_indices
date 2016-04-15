from __future__ import division
import logging
import numpy as np
import thornthwaite
import unittest

# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
# our test case class
class ThornthwaiteTestCase(unittest.TestCase):
    """Tests for `thornthwaite.py`."""
    
    @classmethod
    def setUpClass(cls):
        
        # initialize our month scales and expected results for the various 
        # fitting tests we'll do for each distribution for each month scale
    
        # expected results for PET at latitude 45.0
        cls.fixture_results_pet = np.array([0.0000, 0.0000, 9.4886, 40.8408, 74.3253, 101.7806, 129.5835, 121.5981, 76.9191, 39.4350,
                                            4.3550, 0.0000, 2.3510, 0.9678, 12.2548, 27.9763, 67.1060, 127.8626, 133.9830, 114.8730,
                                            75.5393, 42.5878, 4.3384, 2.7737, 0.0000, 0.0000, 0.0000, 35.3868, 90.4441, 107.0270,
                                            130.5260, 123.0692, 81.2998, 37.4015, 11.6981, 0.0000, 0.0000, 0.0000, 1.7247, 53.0313,
                                            67.5310, 122.5933, 147.9228, 129.9176, 82.3020, 36.9412, 9.3829, 0.0000, 0.0000, 3.1043,
                                            17.2884, 46.2172, 68.2528, 123.3782, 151.0197, 109.1825, 97.5060, 36.9412, 18.8496, 0.0446,
                                            7.8120, 6.3074, 29.0693, 25.2600, 86.0411, 131.4041, 150.9459, 111.9568, 66.2376, 40.7358,
                                            15.5613, 0.0000, 0.0000, 0.9840, 11.0706, 32.2688, 85.8386, 110.9648, 152.0537, 124.8106,
                                            76.6981, 46.8393, 16.3133, 0.0128, 0.0000, 3.5876, 1.9677, 44.3236, 78.5189, 126.1660,
                                            131.7597, 117.9297, 83.3057, 51.7934, 8.9810, 0.0000, 0.0000, 0.0000, 10.9412, 36.4157,
                                            73.9936, 114.4244, 125.8940, 120.3960, 72.3490, 41.1549, 15.5256, 0.0000, 0.0000, 4.2671,
                                            12.2986, 33.1976, 74.7744, 108.4729, 122.6380, 112.7583, 72.5801, 35.0776, 11.8944, 0.0000,
                                            0.0000, 0.0000, 15.4337, 30.8294, 58.1712, 100.0390, 126.2551, 113.6785, 69.6126, 30.8428,
                                            6.2333, 0.0000, 0.0000, 6.6347, 8.3539, 31.0954, 67.7934, 102.8273, 131.3967, 103.6597,
                                            68.1403, 36.4357, 6.7148, 4.3877, 0.0000, 11.5727, 14.3602, 48.0643, 59.0091, 86.2427,
                                            125.1721, 107.9303, 73.5021, 43.5399, 11.2508, 2.8873, 0.0000, 0.4519, 14.2469, 34.8168,
                                            52.2217, 96.3960, 131.8058, 110.6483, 71.5969, 25.6450, 10.1879, 0.0000, 1.7083, 0.0000,
                                            6.9106, 30.7763, 63.0900, 114.7781, 133.5039, 114.0099, 68.7397, 42.1818, 10.0912, 0.0000,
                                            0.0000, 0.0000, 29.7493, 45.6035, 85.7036, 119.1753, 137.8033, 119.8622, 83.0826, 43.6807,
                                            22.2839, 5.2241, 2.3510, 0.0000, 19.3984, 28.2908, 65.5672, 106.4658, 128.4970, 113.7447,
                                            78.9117, 34.3765, 5.4408, 0.0000, 0.0000, 2.2188, 8.3920, 20.3680, 61.6180, 108.0512,
                                            113.9799, 106.3120, 59.6111, 28.5565, 8.1526, 0.0000, 0.0000, 0.0000, 4.1386, 34.1462,
                                            73.9273, 100.4567, 127.1224, 112.6848, 69.7764, 33.2842, 11.3195, 0.0000, 0.0000, 0.0000,
                                            21.4899, 38.2656, 77.8514, 100.1782, 114.6196, 120.7966, 83.0268, 41.5746, 17.7572, 0.0000,
                                            0.0000, 0.0000, 15.4337, 41.1159, 60.3011, 96.9134, 130.0184, 121.3308, 69.7218, 42.7431,
                                            11.9053, 0.0000, 0.0000, 7.2988, 23.9043, 43.3786, 69.7388, 116.5935, 133.5472, 106.0499,
                                            66.5436, 28.2459, 8.4163, 0.0000, 0.0000, 0.0000, 0.2111, 23.7092, 45.9753, 107.7289,
                                            142.1924, 113.8773, 76.3115, 47.2182, 17.5397, 9.2676, 0.0000, 4.7035, 21.3964, 35.2787,
                                            56.6923, 130.0383, 117.0502, 97.2614, 78.1361, 43.8685, 5.4094, 0.0000, 0.0000, 0.0000,
                                            7.4428, 42.5502, 80.5926, 113.5053, 131.2515, 116.9316, 73.3922, 18.8902, 10.8057, 0.0000,
                                            0.5685, 4.0747, 6.8603, 19.3566, 67.7630, 101.7499, 127.3924, 105.9843, 66.7600, 31.0108,
                                            9.3137, 0.0000, 0.0000, 5.5625, 22.5687, 21.5078, 60.0424, 109.2047, 130.9612, 107.1403,
                                            76.2563, 49.7378, 17.3586, 5.4359, 0.0000, 0.0000, 6.2222, 20.1393, 68.1871, 116.1235,
                                            137.2924, 113.6122, 86.3256, 41.3880, 6.6825, 5.2241, 2.3795, 0.0000, 6.2222, 30.2984,
                                            76.1857, 93.3858, 135.1780, 107.6669, 68.4672, 32.0607, 11.8362, 0.0000, 0.0000, 8.2324,
                                            2.1889, 31.8554, 80.2539, 116.8063, 129.1267, 114.4761, 75.1549, 33.9872, 10.5266, 0.0000,
                                            0.0000, 5.2423, 16.6976, 38.4840, 83.9508, 98.7178, 138.4606, 104.1842, 67.3785, 32.9211,
                                            9.5510, 1.8467, 0.0000, 6.2748, 19.4908, 38.0474, 71.0172, 118.4648, 132.0502, 116.9316,
                                            75.6493, 44.9975, 15.6685, 0.0000, 0.6166, 4.9245, 11.1569, 36.5785, 72.8013, 107.6587,
                                            136.9275, 104.9059, 72.0200, 43.2585, 16.8890, 0.0000, 1.4641, 2.1004, 23.6191, 32.0161,
                                            80.9253, 106.5067, 135.9462, 114.6745, 85.4473, 44.4568, 9.7832, 0.0000, 0.0000, 0.0000,
                                            8.4791, 24.6383, 73.4634, 106.9568, 137.3654, 117.5969, 74.1069, 44.2445, 5.0653, 4.6546,
                                            0.0000, 6.7333, 13.3846, 51.4445, 53.3024, 113.1520, 132.9222, 111.4278, 74.5472, 35.9310,
                                            6.1375, 0.0000, 0.0000, 4.7980, 13.4288, 43.0481, 75.8530, 113.0107, 150.8720, 122.3334,
                                            80.2436, 46.2717, 3.4793, 0.0000, 0.0000, 0.0000, 11.8172, 37.3721, 68.6184, 107.0680,
                                            135.8006, 120.5797, 84.3330, 38.1432, 16.6811, 0.0000, 0.0000, 0.0000, 14.4047, 25.4157,
                                            52.4748, 117.3998, 146.5244, 120.1958, 91.9516, 53.2335, 15.0980, 2.9717, 1.4373, 12.2423,
                                            35.5038, 53.5426, 94.3950, 100.5263, 145.1277, 123.9395, 78.9117, 49.7378, 16.5649, 1.3247,
                                            0.4271, 5.9174, 7.9382, 38.4840, 61.9856, 119.4597, 133.5039, 118.1295, 82.9152, 40.2706,
                                            7.0706, 0.0000, 0.0000, 1.5210, 17.2301, 46.2809, 85.4411, 126.8590, 140.3210, 121.0456,
                                            76.8040, 39.7567, 10.0528, 0.0000, 0.0000, 0.0000, 12.7683, 33.5010, 85.9061, 110.2603,
                                            138.1685, 123.1361, 88.5146, 47.8822, 17.7572, 5.1638, 1.3571, 0.0000, 8.2705, 40.0716,
                                            68.5155, 111.5992, 129.8009, 122.0659, 86.8862, 42.4623, 2.4293, 0.6368, 0.0000, 0.0000,
                                            15.2541, 47.2236, 81.4641, 112.2341, 145.1277, 130.3218, 79.4663, 39.4350, 20.5027, 9.8521,
                                            1.0941, 2.7000, 22.3412, 41.3828, 90.4625, 130.6633, 145.9596, 122.6449, 78.1816, 43.7080,
                                            8.0868, 1.3239, 0.0000, 8.5698, 13.9601, 20.8475, 73.3309, 98.6484, 132.9949, 105.6939,
                                            68.9033, 30.8878, 14.2825, 1.2991, 0.0000, 0.0000, 8.3539, 37.3393, 66.2865, 111.2467,
                                            141.8260, 116.7321, 78.7454, 43.5869, 16.5289, 6.4783, 1.5450, 6.4708, 17.8360, 46.4965,
                                            75.2548, 99.0653, 136.5628, 116.4662, 85.5414, 45.7048, 14.1059, 1.1222, 0.0000, 0.0000,
                                            6.1295, 23.2832, 72.7153, 92.7316, 129.6330, 118.7844, 86.2842, 46.7594, 9.6824, 0.9974,
                                            1.9290, 5.1149, 5.0301, 28.6594, 74.5243, 98.7178, 139.4838, 119.9957, 84.1433, 46.2244,
                                            11.2165, 0.0000, 0.0000, 0.1451, 16.4257, 49.4693, 71.0832, 116.2652, 137.2924, 120.1291,
                                            83.7523, 30.4830, 7.5920, 6.2619, 0.0000, 11.0823, 22.3806, 37.0130, 87.8668, 102.4085,
                                            135.9066, 113.2146, 90.9358, 44.7149, 3.4793, 0.0000, 0.7638, 0.0000, 0.0000, 38.8497,
                                            75.7732, 105.3150, 133.9830, 114.4100, 86.0051, 44.5505, 4.9220, 0.0000, 0.0000, 0.0000,
                                            9.9984, 45.8266, 68.2528, 102.9670, 136.1982, 111.6261, 85.2615, 35.4729, 21.8369, 0.0000,
                                            0.0000, 6.1445, 15.4786, 44.4903, 67.0066, 107.9396, 131.6871, 109.4464, 74.7124, 56.6603,
                                            19.6189, 10.8357, 0.0000, 2.1657, 10.5114, 39.7425, 73.4634, 103.7355, 142.4123, 112.1553,
                                            83.6965, 39.1570, 7.4939, 0.0000, 0.0000, 0.0000, 1.2571, 39.2339, 80.5895, 105.0349,
                                            132.9665, 119.9144, 84.7785, 53.9401, 4.1560, 0.0000, 6.5090, 0.4828, 17.2429, 33.7697,
                                            55.2181, 112.1636, 143.6591, 112.5524, 84.3668, 41.7613, 16.2774, 0.0000, 1.4641, 12.3485,
                                            8.0211, 52.2372, 86.3113, 108.0800, 139.7032, 107.6669, 80.3547, 47.0761, 18.7035, 0.0000,
                                            0.0000, 0.0000, 11.1569, 28.8175, 68.5812, 104.0851, 131.5419, 123.8726, 83.1941, 51.3144,
                                            13.5428, 6.4783, 6.9293, 0.0000, 15.9109, 35.3044, 80.1868, 119.7901, 133.8377, 107.8864,
                                            88.4087, 42.5412, 6.0506, 0.3646, 0.0000, 11.5727, 18.7072, 32.3757, 58.2356, 116.1235,
                                            133.5039, 113.8110, 79.4663, 39.5278, 3.6286, 3.7438, 0.0000, 8.3331, 3.5316, 30.2984,
                                            85.5011, 116.1235, 132.7768, 130.7262, 82.8595, 47.5975, 10.3287, 6.4164, 0.0000, 0.0000,
                                            14.0489, 45.4921, 69.4359, 125.3074, 149.6178, 117.1312, 76.4219, 43.3523, 11.5603, 3.0281,
                                            0.0000, 0.0000, 20.8380, 38.9594, 75.8398, 124.5688, 144.2725, 123.6457, 91.3260, 40.0804,
                                            13.1420, 0.0000, 0.0000, 5.1786, 11.4164, 35.7113, 72.9337, 129.3919, 147.6283, 119.4621,
                                            69.0123, 38.5553, 6.8116, 0.0000, 0.0000, 2.4837, 3.0845, 50.7098, 66.6137, 104.3648,
                                            130.1634, 117.5969, 85.1496, 46.8393, 19.6924, 3.5992, 0.0000, 11.1172, 10.1264, 27.3458,
                                            83.4124, 100.8745, 140.8739, 116.9981, 86.1575, 53.7145, 16.0263, 0.0000, 0.0000, 0.0000,
                                            1.8716, 29.0330, 66.0563, 105.1749, 141.4904, 115.6673, 73.3459, 54.9003, 5.8290, 0.8749,
                                            3.8503, 1.3291, 5.9814, 29.4511, 60.0424, 90.0804, 130.4534, 115.0713, 64.8284, 48.8327,
                                            19.1789, 1.0972, 0.0000, 0.0000, 18.9833, 40.6758, 85.9061, 113.6466, 135.6880, 124.8106,
                                            84.6463, 43.1179, 18.3388, 0.0281, 0.0000, 4.7980, 19.5833, 20.4930, 69.2386, 98.3705,
                                            141.9726, 121.5313, 79.0226, 47.1709, 19.2521, 0.0000, 0.0000, 11.4334, 18.2392, 28.6097,
                                            73.3123, 114.6098, 132.6036, 101.7362, 78.5678, 43.9419, 12.1358, 0.0000, 5.2684, 0.0000,
                                            6.3430, 38.1020, 83.1433, 105.2046, 136.7816, 129.9176, 85.3735, 33.0572, 11.2165, 4.4173,
                                            0.2046, 8.9432, 9.1506, 18.8820, 77.3179, 114.9904, 140.2153, 126.8235, 71.1434, 33.7388,
                                            14.6364, 0.3648, 0.0000, 0.9063, 12.1118, 34.0386, 59.2027, 110.6125, 143.1456, 119.5955,
                                            74.1620, 31.3834, 9.3493, 0.0000, 0.0000, 5.8419, 31.6996, 40.4444, 76.9739, 115.2470,
                                            143.9061, 115.7997, 74.7707, 35.1687, 6.3370, 0.0000, 0.0000, 0.0000, 4.5236, 28.4487,
                                            77.9181, 112.1636, 134.9595, 117.9963, 73.2273, 42.2285, 11.2852, 3.2266, 0.0000, 0.0000,
                                            23.7011, 30.6169, 87.4607, 126.8820, 135.1051, 114.0762, 80.5769, 45.7992, 11.6637, 0.0000,
                                            0.0000, 0.0000, 6.1820, 14.3466, 59.1382, 101.8504, 133.3584, 108.7210, 79.8548, 39.4814,
                                            9.2487, 2.2779, 0.0000, 8.0933, 9.1077, 30.2007, 79.9185, 106.9277, 135.1459, 111.9665,
                                            75.4294, 39.1103, 16.0366, 0.1756, 0.0000, 7.1956, 5.8215, 43.4358, 58.7512, 121.3102,
                                            136.9275, 120.4628, 82.8037, 48.9279, 16.2415, 8.1439, 2.4366, 3.3449, 23.4173, 33.8772,
                                            66.4828, 113.8586, 135.3237, 114.8058, 74.9327, 52.6569, 11.4226, 0.0000, 0.0000, 0.0000,
                                            10.3829, 33.7159, 72.0738, 109.3454, 135.1051, 108.7869, 89.3583, 51.8893, 6.2652, 4.0642,
                                            2.9575, 7.4362, 6.4531, 35.6300, 58.8958, 105.3851, 136.6013, 113.8811, 80.3918, 42.1686,
                                            14.8280, 12.4586, 7.8120, 6.6675, 13.2523, 49.5256, 72.2060, 127.5268, 142.9989, 118.4624,
                                            81.2998, 34.4221, 17.4310, 4.9834, 0.0000, 0.4353, 12.3302, 31.4149, 71.3472, 105.1346,
                                            130.9612, 114.1425, 77.5273, 31.5638, 7.9859, 0.0000, 1.0424, 4.8612, 13.3405, 19.4340,
                                            63.4153, 107.9396, 132.8495, 115.0049, 85.3735, 44.8091, 12.6338, 3.0281, 0.0834, 0.9409,
                                            12.2109, 27.2395, 92.3032, 108.7542, 135.6551, 116.7936, 84.8899, 28.8230, 11.2764, 0.0000,
                                            0.0000, 0.0000, 12.7244, 47.0557, 77.1846, 122.6646, 143.5857, 120.7298, 66.2918, 42.1818,
                                            5.8513, 2.5251, 10.5302, 9.6615, 27.0407, 39.3042, 79.2540, 120.9541, 128.2798, 123.2031,
                                            63.4228, 33.6933, 12.9476, 1.6092, 0.0000, 1.4924, 7.9382, 45.8824, 72.9337, 121.2389,
                                            125.3164, 112.3539, 79.7438, 50.4059, 11.1479, 0.0000, 0.0000, 6.6163, 12.7824, 36.5542,
                                            71.7876, 121.6418, 146.7673, 118.5188, 75.9789, 57.4063, 13.2118, 0.0000, 0.0000, 0.0000,
                                            28.8268, 56.5081, 79.5215, 114.8489, 146.4509, 112.2215, 79.7993, 40.2241, 15.8115, 1.9802,
                                            0.0000, 0.0300, 20.6968, 46.2172, 74.3916, 125.8081, 141.3865, 114.4078, 85.8214, 44.9504,
                                            11.3539, 0.0000, 0.0000, 8.7054, 4.9127, 27.9754, 62.1803, 104.9946, 140.8007, 122.5340,
                                            83.0268, 49.6424, 10.6009, 0.0000, 0.0000, 6.7862, 19.1625, 55.0033, 82.7415, 109.3170,
                                            128.8374, 118.5188, 81.7215, 51.3580, 6.4328, 0.0000, 0.0000, 1.5748, 21.6771, 40.2912,
                                            78.4521, 105.0646, 130.2359, 115.0049, 78.2468, 40.3171, 6.5216, 0.0000, 2.8116, 0.0000,
                                            24.8401, 39.6329, 77.9181, 132.6277, 148.4384, 132.5478, 85.3735, 37.5397, 0.5668, 0.0000,
                                            0.0000, 17.6730, 16.5616, 29.2397, 53.8124, 90.9054, 132.7768, 128.1676, 85.2615, 46.8867,
                                            24.4217, 4.6546, 1.2775, 9.9595, 19.7655, 40.1689, 88.0832, 128.7953, 149.2673, 128.0609,
                                            74.1127, 35.4420, 12.7245, 2.0329, 0.0000, 0.6781, 28.3911, 28.9230, 88.0022, 111.8814,
                                            131.6145, 117.1977, 81.8007, 39.1570, 14.2472, 0.0000, 2.7825, 0.0000, 14.2712, 23.5033,
                                            62.6350, 95.6666, 142.6322, 126.5549, 77.8593, 35.3814, 12.6338, 1.4273, 6.8320, 8.3331,
                                            26.2248, 21.9154, 68.6469, 111.7403, 130.7436, 112.3539, 76.9191, 52.1290, 24.9126, 0.0128,
                                            5.2370, 9.3205, 15.8204, 54.0316, 96.6153, 127.1457, 145.5927, 123.9795, 86.1726, 39.8954,
                                            1.9159, 5.9511, 0.0000, 2.9848, 21.6303, 37.1761, 93.0985, 124.2352, 135.5422, 122.8685,
                                            91.8387, 54.7265, 15.4542, 0.0000, 0.0000, 2.8067, 13.6499, 53.3153, 82.4711, 130.7569,
                                            155.1608, 129.3115, 77.9146, 37.7703, 13.6130, 0.0000, 13.4778, 2.0515, 18.7072, 31.4681,
                                            83.0761, 119.6019, 156.9397, 125.6824, 88.4584, 59.4786, 7.8544, 1.9802, 0.0000, 0.0000,
                                            37.2677, 37.5359, 83.0110, 119.0078, 141.4904, 115.6673, 74.7159, 35.5332, 7.1083, 2.2767,
                                            5.7417, 4.5777, 13.9601, 34.1462, 77.5179, 101.5714, 153.6803, 112.4862, 79.6883, 44.3385,
                                            18.8131, 2.9435, 2.0126, 7.1293, 9.4040, 43.0481, 92.1445, 132.4836, 146.5244, 118.9955,
                                            70.3228, 37.8625, 16.8169, 0.6135, 0.0000, 7.5280, 28.1494, 43.7685, 83.7488, 127.7419,
                                            152.9407, 129.1095, 82.8037, 42.1818, 23.9697, 0.0000, 0.0000, 2.5484, 19.4405, 36.4998,
                                            62.9846, 118.5102, 146.3267, 127.1227, 84.0547, 46.9952, 20.6453, 0.0000, 4.6762, 4.9245,
                                            18.8452, 33.3937, 93.7806, 98.1622, 147.6283, 122.4671, 86.3816, 34.6958, 18.8131, 0.0000,
                                            0.0000, 0.6533, 12.6366, 30.9358, 57.8493, 119.1043, 146.7451, 117.4638, 92.3470, 42.6963,
                                            7.3959, 6.4473, 0.0000, 0.0000, 19.1216, 31.1486, 61.7909, 113.0107, 139.8495, 132.9530,
                                            86.0454, 45.3746, 11.5258, 0.0000, 6.2519, 2.1004, 23.4767, 46.3370, 89.1699, 134.0482,
                                            137.3297, 124.7143, 84.8342, 50.1678, 21.2356, 0.0000, 0.0000, 0.0000, 26.9926, 41.1159,
                                            78.1183, 136.0897, 143.5857, 118.0629, 78.1915, 33.8753, 14.0001, 0.0000, 7.8120, 11.5025,
                                            20.3248, 39.5233, 77.8514, 124.8069, 145.5686, 109.2485, 87.4473, 54.7748, 17.6846, 0.9979,
                                            5.1117, 17.0458, 31.2130, 37.9929, 61.5964, 128.1003, 128.5694, 121.3308, 92.2905, 50.4537,
                                            6.5859, 0.0000, 0.0000, 11.7602, 25.0490])
        
        # create input array (from nclimgrid_tavg.nc at lon:300, lat:300)
        cls.fixture_data = np.array([-2.82, -1.22, 2.55, 8.45, 12.75, 16.61, 20.35, 20.71, 15.84, 9.68, 1.62, -3.59,
                                    0.94, 0.41, 3.18, 6.04, 11.63, 20.29, 20.99, 19.76, 15.65, 10.4, 1.62, 1.13,
                                    -1.52, -1.84, -0.39, 7.45, 15.15, 17.36, 20.48, 20.93, 16.63, 9.24, 3.86, -3.21,
                                    -4.87, -0.04, 0.57, 10.63, 11.72, 19.56, 22.86, 21.95, 16.81, 9.14, 3.18, -3.75,
                                    -2.1, 1.18, 4.32, 9.42, 11.83, 19.67, 23.28, 18.84, 19.51, 9.14, 5.87, 0.03,
                                    2.7, 2.2, 6.82, 5.54, 14.5, 20.79, 23.27, 19.26, 13.89, 9.96, 4.96, -0.26,
                                    -1.43, 0.43, 2.92, 6.87, 14.47, 17.92, 23.42, 21.19, 15.8, 11.26, 5.17, 0.01,
                                    -0.19, 1.34, 0.64, 9.08, 13.38, 20.06, 20.65, 20.16, 16.99, 12.3, 3.06, -1.04,
                                    -0.58, -5.39, 2.89, 7.64, 12.7, 18.41, 19.84, 20.53, 15.01, 10.05, 4.95, -1.04,
                                    -2.59, 1.51, 3.19, 7.02, 12.79, 17.56, 19.42, 19.44, 15.11, 8.77, 3.93, -1.09,
                                    -1.54, -1.63, 3.91, 6.6, 10.28, 16.36, 19.89, 19.52, 14.51, 7.8, 2.22, -3.74,
                                    -2.83, 2.3, 2.28, 6.65, 11.76, 16.76, 20.6, 18.0, 14.24, 9.03, 2.37, 1.69,
                                    -0.44, 3.75, 3.67, 9.75, 10.41, 14.36, 19.74, 18.65, 15.22, 10.56, 3.73, 1.17,
                                    -0.13, 0.21, 3.63, 7.32, 9.33, 15.83, 20.69, 19.12, 14.93, 6.66, 3.43, -0.76,
                                    0.71, -0.0, 1.93, 6.59, 11.04, 18.46, 20.89, 19.57, 14.35, 10.27, 3.39, -6.11,
                                    -2.86, -0.67, 6.96, 9.31, 14.45, 19.08, 21.48, 20.45, 16.95, 10.59, 6.8, 1.97,
                                    0.94, -1.99, 4.78, 6.12, 11.42, 17.28, 20.2, 19.53, 16.2, 8.58, 1.97, -3.69,
                                    -1.04, 0.85, 2.28, 4.57, 10.79, 17.5, 18.21, 18.46, 12.71, 7.32, 2.82, -3.47,
                                    -5.76, -1.87, 1.23, 7.22, 12.69, 16.42, 20.01, 19.37, 14.54, 8.34, 3.75, -2.67,
                                    -0.54, -1.07, 5.23, 7.98, 13.28, 16.38, 18.27, 20.59, 16.94, 10.14, 5.57, -2.78,
                                    -2.99, -0.79, 3.91, 8.5, 10.61, 15.91, 20.41, 20.67, 14.53, 10.39, 3.92, -0.17,
                                    -1.91, 2.42, 5.72, 8.88, 12.03, 18.71, 20.93, 18.42, 14.0, 7.25, 2.9, -1.91,
                                    -4.32, -0.6, 0.09, 5.24, 8.36, 17.46, 22.08, 19.55, 15.73, 11.34, 5.51, 3.26,
                                    -0.72, 1.7, 5.21, 7.43, 10.05, 20.6, 18.61, 17.02, 16.06, 10.63, 1.96, -1.25,
                                    -2.29, -1.48, 2.06, 8.76, 13.69, 18.28, 20.58, 20.01, 15.2, 5.07, 3.6, -0.44,
                                    0.27, 1.45, 1.91, 4.37, 11.73, 16.6, 20.08, 18.41, 14.04, 7.87, 3.17, -1.3,
                                    -0.36, 1.97, 5.46, 4.81, 10.57, 17.67, 20.54, 18.53, 15.72, 11.87, 5.46, 2.04,
                                    -3.62, -0.59, 1.76, 4.54, 11.82, 18.65, 21.41, 19.51, 17.53, 10.1, 2.36, 1.97,
                                    0.95, -1.58, 1.76, 6.5, 13.03, 15.4, 21.12, 18.61, 14.3, 8.07, 3.9, -0.88,
                                    -3.2, 2.69, 0.7, 6.77, 13.61, 18.74, 20.32, 19.7, 15.58, 8.53, 3.53, -4.3,
                                    -5.26, 1.87, 4.19, 8.02, 14.19, 16.17, 21.57, 18.08, 14.1, 8.26, 3.23, 0.79,
                                    -1.37, 2.19, 4.8, 7.94, 12.25, 18.98, 20.69, 20.01, 15.61, 10.87, 4.99, -1.72,
                                    0.29, 1.77, 2.94, 7.67, 12.52, 17.45, 21.36, 18.19, 14.95, 10.5, 5.33, -2.51,
                                    0.62, 0.81, 5.66, 6.8, 13.71, 17.28, 21.26, 19.73, 17.44, 10.8, 3.31, -1.98,
                                    -2.29, -3.19, 2.31, 5.42, 12.62, 17.35, 21.42, 20.11, 15.33, 10.71, 1.85, 1.78,
                                    -3.66, 2.33, 3.45, 10.35, 9.52, 18.23, 20.81, 19.18, 15.41, 8.92, 2.19, -2.57,
                                    -1.35, 1.73, 3.46, 8.85, 12.98, 18.21, 23.26, 20.82, 16.44, 11.14, 1.33, -4.2,
                                    -5.14, -0.17, 3.08, 7.79, 11.86, 17.36, 21.24, 20.62, 17.24, 9.44, 5.29, -4.65,
                                    -4.78, -5.48, 3.68, 5.57, 9.39, 18.83, 22.67, 20.5, 18.53, 12.6, 4.83, 1.2,
                                    0.61, 3.94, 8.13, 10.72, 15.73, 16.43, 22.48, 21.06, 16.2, 11.87, 5.24, 0.59,
                                    0.21, 2.08, 2.18, 8.02, 10.87, 19.12, 20.89, 20.19, 16.92, 9.86, 2.48, -0.35,
                                    -0.51, 0.61, 4.29, 9.4, 14.38, 20.15, 21.86, 20.69, 15.88, 9.79, 3.39, -0.02,
                                    -10.0, -1.62, 3.31, 7.1, 14.48, 17.82, 21.53, 20.94, 17.92, 11.48, 5.57, 1.95,
                                    0.58, -0.15, 2.26, 8.31, 11.87, 18.01, 20.38, 20.78, 17.63, 10.33, 0.97, 0.31,
                                    -0.64, -6.25, 3.87, 9.6, 13.82, 18.1, 22.48, 22.01, 16.3, 9.68, 6.32, 3.44,
                                    0.48, 1.01, 5.39, 8.52, 15.12, 20.68, 22.63, 20.93, 16.13, 10.64, 2.8, 0.59,
                                    -0.08, 2.88, 3.58, 4.68, 12.6, 16.16, 20.82, 18.31, 14.38, 7.81, 4.6, 0.58,
                                    -0.99, -2.51, 2.28, 7.81, 11.53, 17.96, 22.03, 19.98, 16.17, 10.57, 5.23, 2.38,
                                    0.65, 2.25, 4.44, 9.47, 12.89, 16.22, 21.31, 19.94, 17.39, 11.02, 4.55, 0.51,
                                    -1.4, -0.61, 1.73, 5.14, 12.48, 15.3, 20.39, 20.35, 17.59, 11.29, 3.28, 0.46,
                                    0.79, 1.83, 1.46, 6.19, 12.78, 16.17, 21.71, 20.47, 17.14, 11.13, 3.72, -2.13,
                                    -1.05, 0.08, 4.13, 10.0, 12.26, 18.67, 21.41, 20.49, 17.07, 7.72, 2.64, 2.31,
                                    -1.45, 3.61, 5.42, 7.75, 14.77, 16.7, 21.22, 19.45, 18.35, 10.81, 1.33, -1.67,
                                    0.35, -1.5, -0.41, 8.06, 12.94, 17.11, 20.99, 19.69, 17.54, 10.82, 1.81, -1.38,
                                    -6.75, -5.27, 2.67, 9.35, 11.83, 16.78, 21.26, 19.21, 17.34, 8.82, 6.68, -1.87,
                                    -2.52, 2.15, 3.92, 9.11, 11.64, 17.49, 20.64, 18.88, 15.44, 13.31, 6.08, 3.74,
                                    -0.87, 0.86, 2.79, 8.25, 12.62, 16.89, 22.11, 19.29, 17.06, 9.62, 2.61, -1.71,
                                    -1.89, -0.13, 0.43, 8.13, 13.66, 17.07, 20.85, 20.52, 17.32, 12.8, 1.56, -0.61,
                                    2.3, 0.23, 4.31, 7.15, 9.82, 18.09, 22.28, 19.35, 17.18, 10.18, 5.16, -1.53,
                                    0.62, 3.97, 2.2, 10.49, 14.54, 17.51, 21.74, 18.61, 16.46, 11.31, 5.83, -0.47,
                                    -3.08, -2.71, 2.94, 6.22, 11.88, 16.94, 20.62, 21.05, 16.97, 12.2, 4.39, 2.38,
                                    2.43, -1.11, 4.0, 7.41, 13.6, 19.16, 20.97, 18.7, 17.97, 10.39, 2.17, 0.19,
                                    -0.43, 3.75, 4.63, 6.89, 10.29, 18.65, 20.89, 19.54, 16.3, 9.7, 1.38, 1.47,
                                    0.0, 2.81, 1.07, 6.5, 14.42, 18.65, 20.79, 22.07, 16.91, 11.42, 3.46, 2.36,
                                    -0.06, -0.23, 3.6, 9.29, 12.01, 19.94, 23.09, 20.04, 15.75, 10.52, 3.82, 1.22,
                                    -3.49, -0.91, 5.07, 8.08, 12.95, 19.83, 22.4, 21.08, 18.49, 9.86, 4.29, -1.23,
                                    -0.24, 1.85, 3.0, 7.51, 12.54, 20.51, 22.82, 20.39, 14.4, 9.49, 2.4, -3.29,
                                    -1.82, 0.97, 0.95, 10.22, 11.58, 16.98, 20.43, 20.11, 17.32, 11.26, 6.1, 1.42,
                                    -3.9, 3.62, 2.7, 5.94, 14.11, 16.48, 21.9, 20.02, 17.5, 12.7, 5.09, -0.46,
                                    -2.27, -1.99, 0.61, 6.24, 11.47, 17.09, 22.02, 19.88, 15.25, 13.0, 2.1, 0.41,
                                    1.45, 0.56, 1.7, 6.34, 10.57, 14.92, 20.47, 19.73, 13.63, 11.68, 5.96, 0.5,
                                    -1.41, -2.94, 4.69, 8.42, 14.48, 18.3, 21.19, 21.19, 17.23, 10.47, 5.73, 0.02,
                                    -1.84, 1.73, 4.82, 4.61, 11.98, 16.12, 22.05, 20.7, 16.22, 11.33, 5.98, -4.21,
                                    -2.97, 3.59, 4.51, 6.16, 12.57, 18.43, 20.8, 17.76, 16.2, 10.69, 4.0, -2.49,
                                    1.91, -0.68, 1.79, 7.95, 14.07, 17.1, 21.34, 21.95, 17.36, 8.29, 3.72, 1.7,
                                    0.11, 2.99, 2.47, 4.29, 13.2, 18.49, 21.81, 21.49, 14.79, 8.44, 4.7, 0.19,
                                    -0.85, 0.4, 3.16, 7.2, 10.44, 17.87, 22.21, 20.41, 15.34, 7.92, 3.17, -2.4,
                                    -0.11, 1.99, 7.33, 8.35, 13.12, 18.52, 22.35, 19.9, 15.51, 8.79, 2.26, -2.05,
                                    -4.82, -0.17, 1.33, 6.15, 13.29, 18.09, 21.09, 20.17, 15.17, 10.28, 3.74, 1.29,
                                    -2.43, -0.76, 5.7, 6.56, 14.71, 20.16, 21.11, 19.58, 16.5, 11.04, 3.85, -0.86,
                                    -1.45, -0.27, 1.75, 3.37, 10.43, 16.62, 20.87, 18.77, 16.37, 9.69, 3.14, 0.95,
                                    -0.04, 2.65, 2.45, 6.46, 13.56, 17.34, 21.15, 19.32, 15.63, 9.65, 5.11, 0.1,
                                    -1.16, 2.47, 1.66, 8.92, 10.37, 19.38, 21.36, 20.54, 16.9, 11.7, 5.15, 2.91,
                                    0.97, 1.26, 5.64, 7.17, 11.56, 18.33, 21.14, 19.69, 15.48, 12.48, 3.78, -3.31,
                                    -4.23, -2.63, 2.76, 7.14, 12.41, 17.69, 21.11, 18.78, 18.07, 12.32, 2.23, 1.58,
                                    1.15, 2.46, 1.81, 7.47, 10.37, 17.12, 21.35, 19.61, 16.53, 10.31, 4.77, 4.23,
                                    2.7, 2.31, 3.42, 10.01, 12.43, 20.25, 22.19, 20.24, 16.63, 8.59, 5.48, 1.89,
                                    -1.22, 0.21, 3.21, 6.71, 12.3, 17.09, 20.54, 19.59, 15.95, 7.96, 2.76, -0.59,
                                    0.46, 1.75, 3.44, 4.4, 11.09, 17.49, 20.8, 19.72, 17.36, 10.83, 4.13, 1.22,
                                    0.05, 0.4, 3.17, 5.9, 15.39, 17.6, 21.22, 20.05, 17.34, 7.38, 3.75, -0.27,
                                    -0.98, -0.69, 3.3, 9.57, 13.18, 19.57, 22.27, 20.58, 13.9, 10.27, 2.1, 1.04,
                                    3.51, 3.2, 6.4, 8.17, 13.49, 19.33, 20.17, 20.95, 13.37, 8.43, 4.22, 0.7,
                                    -1.54, 0.62, 2.18, 9.36, 12.54, 19.37, 19.76, 19.32, 16.35, 12.01, 3.7, -1.36,
                                    -1.07, 2.22, 3.3, 7.64, 12.34, 19.42, 22.74, 20.31, 15.73, 13.52, 4.31, -1.19,
                                    -4.03, -0.63, 6.77, 11.24, 13.53, 18.47, 22.66, 19.3, 16.36, 9.85, 5.03, 0.84,
                                    -0.16, 0.02, 5.06, 9.42, 12.76, 20.01, 21.97, 19.63, 17.44, 10.86, 3.76, -3.93,
                                    -1.7, 2.92, 1.43, 6.06, 10.9, 17.07, 21.89, 20.85, 16.94, 11.85, 3.54, -0.29,
                                    -2.42, 2.27, 4.71, 10.94, 13.98, 17.68, 20.28, 20.31, 16.77, 12.26, 2.29, -2.74,
                                    -0.12, 0.65, 5.27, 8.35, 13.37, 17.08, 20.44, 19.72, 16.08, 9.87, 2.31, -0.32,
                                    1.1, -0.74, 5.94, 8.23, 13.29, 20.96, 22.93, 22.34, 17.36, 9.27, 0.27, -0.02,
                                    -0.51, 5.44, 4.16, 6.3, 9.6, 15.04, 20.79, 21.69, 17.34, 11.27, 7.37, 1.78,
                                    0.55, 3.18, 4.84, 8.3, 14.77, 20.42, 23.08, 21.74, 15.39, 8.85, 4.17, 0.86,
                                    -0.22, 0.31, 6.68, 6.24, 14.79, 18.05, 20.63, 20.05, 16.72, 9.62, 4.59, -1.04,
                                    1.09, -0.87, 3.65, 5.2, 10.97, 15.73, 22.14, 21.45, 16.01, 8.8, 4.13, 0.63,
                                    2.4, 2.81, 6.23, 4.89, 11.89, 18.03, 20.51, 19.32, 15.84, 12.37, 7.5, 0.01,
                                    1.9, 3.0, 3.98, 10.77, 16.02, 20.19, 22.58, 21.13, 17.57, 9.82, 0.79, 2.21,
                                    -0.9, 1.14, 5.26, 7.78, 15.54, 19.79, 21.17, 20.9, 18.51, 12.91, 4.93, -1.35,
                                    -0.66, 1.08, 3.51, 10.68, 13.97, 20.7, 23.84, 21.86, 16.02, 9.32, 4.41, -0.55,
                                    4.36, 0.82, 4.63, 6.72, 14.06, 19.14, 24.08, 21.32, 17.91, 13.89, 2.72, 0.84,
                                    -0.71, -1.02, 8.45, 7.82, 14.02, 19.05, 22.02, 19.88, 15.5, 8.87, 2.5, 0.95,
                                    2.06, 1.66, 3.58, 7.22, 13.23, 16.58, 23.64, 19.34, 16.34, 10.73, 5.86, 1.19,
                                    0.82, 2.45, 2.53, 8.85, 15.4, 20.94, 22.67, 20.32, 14.64, 9.34, 5.31, 0.3,
                                    -1.8, 2.57, 6.63, 8.98, 14.16, 20.28, 23.54, 21.83, 16.9, 10.27, 7.25, -1.77,
                                    -2.53, 0.96, 4.77, 7.63, 11.0, 18.98, 22.68, 21.6, 17.19, 11.34, 6.38, -0.89,
                                    1.72, 1.77, 4.66, 7.08, 15.64, 16.09, 22.82, 20.84, 17.54, 8.65, 5.86, -2.06,
                                    -0.26, 0.3, 3.28, 6.62, 10.23, 19.07, 22.7, 20.09, 18.6, 10.38, 2.58, 2.37,
                                    -0.12, -0.65, 4.72, 6.66, 10.84, 18.21, 21.76, 22.4, 17.48, 10.95, 3.81, -0.29,
                                    2.22, 0.81, 5.63, 9.41, 14.93, 21.15, 21.45, 21.24, 17.33, 12.01, 6.54, -0.52,
                                    -2.45, -0.6, 6.39, 8.5, 13.32, 21.44, 22.27, 20.18, 16.07, 8.47, 4.52, -1.15,
                                    2.7, 3.73, 4.98, 8.21, 13.28, 19.87, 22.54, 18.85, 17.73, 12.92, 5.55, 0.46,
                                    1.86, 5.27, 7.26, 7.93, 10.81, 20.33, 20.21, 20.67, 18.59, 12.02, 2.33, -1.46,
                                    -0.67, 3.68, 5.96])

        cls.latitude = 45.0
        cls.fill_value = np.NaN
        
    def test_thornthwaite(self):
    
        logger.info('Testing the PET calculation using Thornthwaite\'s equation')
        pet = thornthwaite.thornthwaite(self.fixture_data, self.latitude, 1895, self.fill_value)
        assert np.allclose(pet, self.fixture_results_pet, atol=0.001, rtol=0.000, equal_nan=True) == True, \
               'One or more of the results did not match within the specified tolerance'

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()