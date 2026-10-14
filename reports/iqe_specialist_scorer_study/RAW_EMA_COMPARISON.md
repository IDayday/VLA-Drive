# Raw and EMA endpoint evidence

| Cell | Update | Endpoint | Expert alone | Oracle with Base | Oracle gain | H_val direct delta | EMA original-Base weight |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E00 | 1000 | ema | 0.916158 | 0.921805 | 0.00789043 | 0.00124342 | 0.367695 |
| E00 | 1000 | raw | 0.897773 | 0.927882 | 0.0139675 | -0.000171876 | 0 |
| E00 | 3000 | ema | 0.887597 | 0.929114 | 0.0151991 | -0.000973843 | 0.0497124 |
| E00 | 3000 | raw | 0.874349 | 0.929388 | 0.0154733 | -0.000269786 | 0 |
| E00 | 5000 | ema | 0.871748 | 0.931275 | 0.0173603 | -0.000872189 | 0.00672111 |
| E00 | 5000 | raw | 0.870653 | 0.93161 | 0.0176954 | -0.00157691 | 0 |
| E01 | 1000 | ema | 0.911144 | 0.923004 | 0.0090896 | -0.000638746 | 0.367695 |
| E01 | 1000 | raw | 0.886994 | 0.932056 | 0.0181413 | -0.00556275 | 0 |
| E01 | 3000 | ema | 0.870457 | 0.932288 | 0.0183737 | -0.00275903 | 0.0497124 |
| E01 | 3000 | raw | 0.848156 | 0.92936 | 0.015445 | -0.0045729 | 0 |
| E01 | 5000 | ema | 0.848756 | 0.932221 | 0.0183059 | -0.00214885 | 0.00672111 |
| E01 | 5000 | raw | 0.843849 | 0.932933 | 0.0190188 | -0.00339988 | 0 |
| E10 | 1000 | ema | 0.86828 | 0.937393 | 0.0234785 | 0.0122639 | 0.367695 |
| E10 | 1000 | raw | 0.698122 | 0.935404 | 0.0214897 | 0.0154405 | 0 |
| E10 | 3000 | ema | 0.705471 | 0.935107 | 0.0211923 | 0.0139672 | 0.0497124 |
| E10 | 3000 | raw | 0.691947 | 0.935336 | 0.0214218 | 0.0168763 | 0 |
| E10 | 5000 | ema | 0.688666 | 0.935092 | 0.0211778 | 0.0135703 | 0.00672111 |
| E10 | 5000 | raw | 0.68582 | 0.934516 | 0.0206015 | 0.0136796 | 0 |
| E11 | 1000 | ema | 0.883171 | 0.936545 | 0.0226303 | 0.0126709 | 0.367695 |
| E11 | 1000 | raw | 0.753639 | 0.937458 | 0.0235438 | 0.0152941 | 0 |
| E11 | 3000 | ema | 0.777063 | 0.940837 | 0.0269226 | 0.0116509 | 0.0497124 |
| E11 | 3000 | raw | 0.750578 | 0.939758 | 0.0258431 | 0.0113278 | 0 |
| E11 | 5000 | ema | 0.751882 | 0.939728 | 0.0258137 | 0.0105214 | 0.00672111 |
| E11 | 5000 | raw | 0.748352 | 0.939771 | 0.025856 | 0.00963681 | 0 |

Rows are stage_val. The main factorial interpretation uses fixed update5000 raw/EMA; system endpoints may be selected from1000/3000/5000 using the locked stage_val rule. Raw and EMA are two snapshots of one expert and are never both added as separate deployment experts. Early EMA retains substantial original Base weight; the column reports decay**updates.

Generated 2026-10-14T07:26:31.328995+00:00. Implementation `0e0ed8d2964c5215d7fcfa712e3f5ab1121ab24d`. Base step 70000, SHA256 `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`. All current scores use NAVSIM v1.1 fixed official reference; internal score unit [0,1]. Artifact root: `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`.
