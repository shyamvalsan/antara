# Antara

A gallery for work that stays with you

Antara is a walkable gallery: you move through a long hall and 6 rooms, one for each artist, and stand in front of 22 pictures, each with a short text, a recording and the program that made it.

It is entirely imagined. The gallery, its curator, the six artists and their lives, the exhibition texts, the building and every picture were conceived and made by Fable 5.2, an AI model, as an experiment by Shyam Sreevalsan. The artists are fictional. The places and trades they come from are real.

Live: **https://shyamvalsan.github.io/antara/**

## How the pictures are made

Every artwork is one self-contained Python program that computes the picture. There is no image-generation model and there are no source images, photographs or textures: each program builds its own geometry, traces its own light and simulates its own paint and ink, and writes the finished image pixel by pixel. The program beside each work in the gallery is the whole of it.

## The works

| Work | Artist | Source | Lines |
| --- | --- | --- | ---: |
| Hanle, three hours, the pole keeps still | Stanzin Dolkar | [hanle_three_hours_the_pole_keeps_still.py](code/dolkar/hanle_three_hours_the_pole_keeps_still.py) | 1,343 |
| Tso Moriri, new moon, the river comes down to the water | Stanzin Dolkar | [tso_moriri_new_moon_the_river_comes_down_to_the_water.py](code/dolkar/tso_moriri_new_moon_the_river_comes_down_to_the_water.py) | 966 |
| Udo channel, kelp, the current turning | Go Sun-deok | [udo_channel_kelp_the_current_turning.py](code/sundeok/udo_channel_kelp_the_current_turning.py) | 777 |
| Twelve metres, the abalone wall, breath nearly gone | Go Sun-deok | [twelve_metres_the_abalone_wall_breath_nearly_gone.py](code/sundeok/twelve_metres_the_abalone_wall_breath_nearly_gone.py) | 890 |
| Winter 1971, a whale went under me | Go Sun-deok | [winter_1971_a_whale_went_under_me.py](code/sundeok/winter_1971_a_whale_went_under_me.py) | 917 |
| Under the float, looking up | Go Sun-deok | [under_the_float_looking_up.py](code/sundeok/under_the_float_looking_up.py) | 755 |
| Kund, three thousand five hundred steps, one going down | Ruhi Vaghela | [kund_three_thousand_five_hundred_steps_one_going_down.py](code/ruhi/kund_three_thousand_five_hundred_steps_one_going_down.py) | 935 |
| Vav, the seventh storey down, noon in May | Ruhi Vaghela | [vav_the_seventh_storey_down_noon_in_may.py](code/ruhi/vav_the_seventh_storey_down_noon_in_may.py) | 1,350 |
| Sala de espera, Aduana, 25:00 | Halina Mróz Quispe | [sala_de_espera_aduana_2500.py](code/halina/sala_de_espera_aduana_2500.py) | 890 |
| Escalera 3, marea alta | Halina Mróz Quispe | [escalera_3_marea_alta.py](code/halina/escalera_3_marea_alta.py) | 1,213 |
| Salar, puerta, mediodía | Halina Mróz Quispe | [salar_puerta_mediodia.py](code/halina/salar_puerta_mediodia.py) | 739 |
| Third Mainland, inbound, 6:40 am, four seconds | Tobi Adeyemi | [third_mainland_inbound_640am_four_seconds.py](code/tobi/third_mainland_inbound_640am_four_seconds.py) | 1,282 |
| Ojuelegba, 6:42 pm, eleven seconds | Tobi Adeyemi | [ojuelegba_642pm_eleven_seconds.py](code/tobi/ojuelegba_642pm_eleven_seconds.py) | 1,371 |
| Oshodi under bridge, rain coming, nine seconds | Tobi Adeyemi | [oshodi_under_bridge_rain_coming_nine_seconds.py](code/tobi/oshodi_under_bridge_rain_coming_nine_seconds.py) | 1,494 |
| Kavu, 6:10 pm, first rain | Ilanji | [kavu_610pm_first_rain.py](code/ilanji/kavu_610pm_first_rain.py) | 1,119 |
| The Nila at Thirunavaya, noon, cloth drying | Ilanji | [nila_thirunavaya_noon_cloth_drying.py](code/ilanji/nila_thirunavaya_noon_cloth_drying.py) | 1,243 |
| Kumano, the fourth bend, komorebi | Ilanji | [kumano_fourth_bend_komorebi.py](code/ilanji/kumano_fourth_bend_komorebi.py) | 1,287 |
| Kotka, 3 pm in December | Ilanji | [kotka_3pm_december.py](code/ilanji/kotka_3pm_december.py) | 1,265 |
| Thrikkarthika, the temple tank, every step a lamp | Ilanji | [thrikkarthika_the_temple_tank_every_step_a_lamp.py](code/ilanji/thrikkarthika_the_temple_tank_every_step_a_lamp.py) | 1,784 |
| Bumthang, a weaver, one butter lamp | Ilanji | [bumthang_a_weaver_one_butter_lamp.py](code/ilanji/bumthang_a_weaver_one_butter_lamp.py) | 1,365 |
| Above Kibber, before the sun reaches the pass | Ilanji | [above_kibber_before_the_sun.py](code/ilanji/above_kibber_before_the_sun.py) | 1,527 |
| Kavu, year unknown, the machines have gone quiet | Ilanji | [kavu_year_unknown_machines_quiet.py](code/ilanji/kavu_year_unknown_machines_quiet.py) | 1,425 |

## Run a picture yourself

You need Python 3.10 or later.

    pip install numba numpy pillow scikit-image scipy
    python hanle_three_hours_the_pole_keeps_still.py

Each program writes its picture (a `.png`) next to itself. They are compute-heavy, and numba compiles on the first run, so be patient. Most also take a scale such as `0.5` for a quicker, smaller study; the top of each file says how.

## Where things are

- `index.html`, `gallery.js`, `community.js`, `style.css`: the 3D gallery (three.js, in `vendor/`).
- `code/<artist>/<work>.py`: the program for each picture, also as a readable page beside it.
- `works/`, `about/`, `guestbook/`: the same content as plain pages.
- `img/`: each picture at several sizes. `audio/`, `music/`: the synthesized audio guide and music.
- `s/`: small pages that give each work and room a share card.
- `source/`: the source of the gallery itself: `build.py` (makes this site), `narrate.py` and `ambient.py` (audio guide and music), `curator/programme.toml` (every text), `web/` (the page), and `backend/` (the small Cloudflare Worker behind hearts, the visitor count and the guest book).

## Licence

Artworks, texts and code © 2026 Shyam Sreevalsan. You may read, run and learn from the code; please ask before reusing the artworks or texts.

Contact: shyam.valsan@gmail.com
