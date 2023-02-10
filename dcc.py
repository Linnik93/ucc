import sys

import base64
import PySimpleGUI as sg
import os
import CustomLogger as cl
import correct
from correct import correct_image, analyze_video, process_video

progress_val = ''
current_in_filename = ""

with open("./logo/logo UCC.png", "rb") as img_file:
    LOGO = base64.b64encode(img_file.read())

IMAGE_TYPES = (".png", ".jpeg", ".jpg", ".bmp")
VIDEO_TYPES = (".mp4", ".mkv", ".avi")

sg.theme('SandyBeach')
sg.set_options(font=("Arial", 13))
sg.set_global_icon(LOGO)

selected_item_path=''
list_of_files_size = 23

list_box_selected_item =0

TempDirPath = os.path.expanduser('~/Documents').replace('\\',"/")+"/UCC/"

image_settings_section = [
    [
        sg.Text(text="Image Settings", size=(34, 1), font=('Arial', 15), justification='center', visible=True, background_color='navajowhite2')
    ],
    [
        sg.Frame('Underwater restoration settings',[
            [
                sg.CBox(text="Manual colors level      ", key="__COLOR_BALANCE_CB__", enable_events=True, font=('Arial', 10)),
                sg.Push(),
                sg.Slider(range=(0, 6), default_value=1, resolution=.01, size=(19, 10), orientation='h', font=('Arial', 10), key="__COLOR_BALANCE_SLIDER__", disabled=True, enable_events=True),
            ],
            [
                sg.CBox(text="Underwater rest level    ", key = "__UNDERWATER_RESTORATION_BLUE_LEVEL_CB__", enable_events=True,font=('Arial', 10)),
                sg.Push(),
                sg.Slider(range=(0.0, 2), default_value=0.8, resolution=.01, size=(19, 10), orientation='h', font=('Arial', 10), key="__UNDERWATER_RESTORATION_BLUE_LEVEL_SLIDER__", disabled=True, enable_events=True),
            ]
        ],border_width=3)
    ],


    [
        sg.Frame('Background settings', [
            [
                sg.CBox(text="White balance", key="__WHITE_BALANCE_CB__", enable_events=True,
                        font=('Arial', 10),
                        default=False),
                sg.Push(),
                sg.Slider(range=(-5, 5), default_value=0, resolution=0.1, size=(19, 10), orientation='h',
                          font=('Arial', 10),
                          key="__WHITE_BALANCE_SLIDER__", disabled=True, enable_events=True)
            ],

            [
                sg.CBox(text="Manual saturation level", key="__SATURATION_CB__", enable_events=True,font=('Arial', 10)),
                sg.Push(),
                sg.Slider(range=(0.0, 2), default_value=1.0, resolution=.1, size=(19, 10), orientation='h', font=('Arial', 10), key="__SATURATION_SLIDER__", disabled=True, enable_events=True)
            ],
            [
                sg.CBox(text="Manual gamma level", key="__GAMMA_CB__", enable_events=True, font=('Arial', 10)),
                sg.Push(),
                sg.Slider(range=(0.0, 3), default_value=1, resolution=.1, size=(19, 10), orientation='h',
                  font=('Arial', 10), key="__GAMMA_SLIDER__", disabled=True, enable_events=True)
            ],
            [
                sg.CBox(text="Manual denoising level", key="__DENOISING_CB__", enable_events=True, font=('Arial', 10)),
                sg.Push(),
                sg.Slider(range=(0, 50), default_value=0, resolution=.1, size=(19, 10), orientation='h', font=('Arial', 10), key="__DENOISING_SLIDER__", disabled=True, enable_events=True)
            ],

            [
                sg.CBox(text="Manual contrast level", key="__CONTRAST_CB__", enable_events=True, font=('Arial', 10)),
                sg.Push(),
                sg.Slider(range=(0, 3), default_value=1, resolution=.1, size=(19, 10), orientation='h', font=('Arial', 10),
                  key="__CONTRAST_SLIDER__", disabled=True, enable_events=True)
            ],
            [
                sg.CBox(text="Manual brightness level", key="__BRIGHTNESS_CB__", enable_events=True, font=('Arial', 10)),
                sg.Push(),
                sg.Slider(range=(0, 3), default_value=1.0, resolution=0.01, size=(19, 10), orientation='h', font=('Arial', 10),
                  key="__BRIGHTNESS_SLIDER__", disabled=True, enable_events=True)
            ],
            [
                sg.CBox(text="Manual sharpness level", key="__SHARPNESS_CB__", enable_events=True, font=('Arial', 10)),
                sg.Push(),
                sg.Slider(range=(-5, 5), default_value=1.0, resolution=0.01, size=(19, 10), orientation='h',
                  font=('Arial', 10),
                  key="__SHARPNESS_SLIDER__", disabled=True, enable_events=True)
                ]
            ], border_width=3)],

    [
        sg.Frame('Color levels', [
            [
                sg.CBox(text="Manual red level         ", key="__ADJ_RED_LEVEL_CB__", enable_events=True, font=('Arial', 10)),
                sg.Push(),
                sg.Slider(range=(0, 3), default_value=1.0, resolution=0.01, size=(19, 10), orientation='h',
                  font=('Arial', 10),
                  key="__ADJ_RED_LEVEL_SLIDER__", disabled=True, enable_events=True)
            ],
            [
                sg.CBox(text="Manual green level       ", key="__ADJ_GREEN_LEVEL_CB__", enable_events=True, font=('Arial', 10)),
                sg.Push(),
                sg.Slider(range=(0, 3), default_value=1.0, resolution=0.01, size=(19, 10), orientation='h',
                  font=('Arial', 10),
                  key="__ADJ_GREEN_LEVEL_SLIDER__", disabled=True, enable_events=True)
            ],
            [
                sg.CBox(text="Manual blue level       ", key="__ADJ_BLUE_LEVEL_CB__", enable_events=True, font=('Arial', 10)),
                sg.Push(),
                sg.Slider(range=(0, 3), default_value=1.0, resolution=0.01, size=(19, 10), orientation='h',
                  font=('Arial', 10),
                  key="__ADJ_BLUE_LEVEL_SLIDER__", disabled=True, enable_events=True)
            ],
            [
                sg.Text(text="", font=('Arial', 5))
            ]
            ] , border_width=3)],
    [

            sg.Listbox(values="", enable_events=True, size=(58, 4), key="__PREVIEW_STATUS__",select_mode='multiple',font=('Arial', 8))

    ],
    [
        sg.Button(button_text="REFRESH PREVIEW", enable_events=True, pad=(4, 3), button_color='sandybrown', key="__REFRESH_PREVIEW__",size = (40,1))
    ],
    [
        sg.Button(button_text="RESTORE DEFAULT SETTINGS", enable_events=True, pad=(4, 3), button_color='sandybrown',
                  key="__RESTORE_DEFAULT_SETTINGS__", size=(40, 1))
    ],
    [
        sg.Button(button_text="SAVE FOR CURRENT ITEM", enable_events=True, pad=(4, 3), button_color='sandybrown',
                  key="__CORRECT_SINGLE__", size=(40, 1))
    ],

]

left_column = [
    [
        sg.FilesBrowse(button_text="Select photos and videos", enable_events=True, key='__INPUT_FILES__', size=(52, 1),button_color='sandybrown')
    ],
    [
    sg.Frame('Preview settings', [
        [
        sg.Text(text="Show video frame on second:       ",font=('Arial', 10)),
        sg.InputText(default_text=2, size=(5, 1), enable_events=True, readonly=False, key="__PREVIEW_FRAME_SECOND__",disabled_readonly_background_color="darkgray"),
        sg.Text(text="                 ", font=('Arial', 5)),
        sg.Button(button_text=" < ", enable_events=True, pad=(4, 0), button_color='sandybrown', key="__PREVIEW_PREVIOUS_FRAME__",size = (10,1),font=('Arial', 8)),
        sg.Button(button_text=" > ", enable_events=True, pad=(4, 0), button_color='sandybrown', key="__PREVIEW_NEXT_FRAME__",size = (10,1),font=('Arial', 8)),
    ],
    [
        sg.CBox(text="Enable preview", key="__PREVIEW_CB__", enable_events=True, font=('Arial', 10), default=True),
        sg.Text(text="                                            ", font=('Arial', 10)),
        sg.CBox(text="Show image settings", key="__IMG_SETTINGS_CB__", enable_events=True, font=('Arial', 10), disabled = True),
    ]
    ], border_width=3)],

        [
            sg.Listbox(values=[], enable_events=True, size=(66, list_of_files_size), key="__INPUT_FILE_LIST__",select_mode='multiple',font=('Arial', 10))
        ],

        [
            sg.Frame('File settings', [
            [
            sg.CBox(text="Temp. folder", key = "__TEMP_FOLDER_CB__", enable_events=True),
            sg.Text("", size=(4, 1)),
            sg.InputText(default_text=TempDirPath, size=(21, 1), enable_events=True, readonly=False, key="__TEMP_FOLDER__",disabled=True,disabled_readonly_background_color="darkgray"),
            sg.FolderBrowse(size=(8, 1),pad=(6,1),disabled=True,key = "__TEMP_FOLDER_BROWSE_BUTTON__",button_color='sandybrown')
        ],
        [
            sg.CBox(text="Output folder     ", key = "__OUTPUT_FOLDER_CB__", enable_events=True),
            sg.Text("", size=(1, 1)),
            sg.InputText(default_text="./", size=(21, 1), enable_events=True, readonly=False, key="__OUTPUT_FOLDER__",disabled=True,disabled_readonly_background_color="darkgray"),
            sg.FolderBrowse(size=(8, 1),pad=(6,1),disabled=True,key = "__OUT_FOLDER_BROWSE_BUTTON__",button_color='sandybrown')
        ],
        [
            sg.CBox(text="Output file prefix", key = "__OUTPUT_PREFIX_CB__", enable_events=True,default=True),
            sg.Text(text="", size=(1, 1)),
            sg.InputText(default_text="corrected", size=(21, 1), key="__OUTPUT_PREFIX__",disabled=False,disabled_readonly_background_color="darkgray")
        ]
        ], border_width=3)],

        [
            sg.Text(text="", font=('Arial', 1))
        ],
    [
        sg.Button(button_text="Correct All", enable_events=True, pad=(2, 5), button_color='sandybrown', key="__CORRECT__",size = (10,1)),
        sg.Button(button_text="Cancel", enable_events=True, pad=(94, 5), disabled=True, key="__CANCEL__", button_color='sandybrown',size = (10,1)),
        sg.Button(button_text="Clear", enable_events=True, pad=(0, 5), disabled=False, key="__CLEAR_LIST__", button_color='sandybrown',size = (10,1))
    ],
    [
        sg.Text(text="", font=('Arial', 1))
    ],
    [   sg.Text( text="STATUS: ", text_color='black'),
        sg.Text( text="", size=(43, 1), text_color='black', key="__STATUS__", background_color='darkgray',)
    ]
]


video_viewer = [
    [
        sg.Text(text="", size=(54, 1), font=('Arial', 15), key="__VIDEO_NAME__", justification='center', visible=True,background_color='navajowhite2')
    ],
    [   sg.Frame('Processing preview', [
        [
        sg.Image(key="__VIDEO_PREVIEW__")
    ]
    ], border_width=3)],
    [
        sg.Text(" ", font=('Arial', 15), key="__PROGBAR_BR__"),
    ],
    [sg.Frame('Progress', [
        [
        sg.Text("Color restoration progress:", size=(36, 1), font=('Arial', 15), key="__PROGBAR_LABEL__",justification='left'),
        sg.Text(text="0%", font=('Arial', 15), size=(14, 1), key="__PROGBAR_PERCENTS__",justification='right')

    ],

    [
        sg.ProgressBar(100, orientation='h', size=(44, 20), key="__PROGBAR__")
    ],

    [
        sg.Text(" ", font=('Arial', 15)),
    ],
    [
        sg.Text("Audio extraction progress:", size=(36, 1), font=('Arial', 15), key="__SOUND_EX_PROGBAR_LABEL__",
                justification='left'),
        sg.Text(text="0%", font=('Arial', 15), size=(14, 1), key="__SOUND_EX_PROGBAR_PERCENTS__",
                justification='right')

    ],
    [
        sg.ProgressBar(100, orientation='h', size=(44, 20), key="__SOUND_EX_PROGBAR__")
    ],

    [
        sg.Text(" ", font=('Arial', 15)),
    ],
    [
        sg.Text("Video encoding progress:", size=(36, 1), font=('Arial', 15), key="__SOUND_PROGBAR_LABEL__",
                justification='left'),
        sg.Text(text="0%", font=('Arial', 15), size=(14, 1), key="__SOUND_PROGBAR_PERCENTS__",
                justification='right')

    ],
    [
        sg.ProgressBar(100, orientation='h', size=(44, 20), key="__SOUND_PROGBAR__")
    ]
    ], border_width=3)],
]

photo_viewer = [
    [
        sg.Text(text="", size=(54, 1), font=('Arial', 15), key="__PHOTO_NAME__", justification='center',visible=True,background_color='navajowhite2')
    ],

    [
        sg.Frame('Initial image', [
    [
        sg.Image( key="__PREVIEW_BEFORE__",size=(576,324))
]
    ], border_width=3)],

    [
        sg.Frame('Final image', [
        [

        sg.Image(key="__PREVIEW_AFTER__",size=(576,324))
]
    ], border_width=3)]

]

layout = [
    [
        sg.Column(left_column,size=(485, 750)),
        sg.Column(video_viewer, key='__VIDEO_VIEWER__',visible=False, vertical_alignment="top", justification="center",size=(598, 750)),
        sg.Column(photo_viewer, key='__PHOTO_VIEWER__',visible=False, vertical_alignment="top", justification="center",size=(598,750)),
        sg.Column(image_settings_section, key='__IMG_SETTINGS__',visible=False,size=(370, 750))
    ]
]

#window = sg.Window("UCC: Underwater Color Corrector", layout, resizable=True, finalize=True)
# Fix scalling level
sg.set_options(scaling=1.333333333)

window = sg.Window("UCC: Underwater Color Corrector", layout, finalize=True)

window.bind('<Configure>',"Window_Event")

def valid_file(path):
    extension = path[path.rfind("."):].lower()
    return os.path.isfile(path) and (extension in IMAGE_TYPES or extension in VIDEO_TYPES)


def get_files(filepaths):
    input_filepaths = [f for f in filepaths if valid_file(f)]

    for f in input_filepaths:
        yield f


file_generator = None
file_index = 0
analyze_video_generator = None
process_video_generator = None

if __name__ == "__main__":


    while True:
        event, values = window.read(1)

        if event == sg.WIN_CLOSED:
            break

        if event == "__PREVIEW_CB__":
            if (values["__PREVIEW_CB__"] == True):
                window["__PREVIEW_FRAME_SECOND__"].update(disabled=False)
                if(len([x for x in window["__INPUT_FILE_LIST__"].get_list_values()])!=0):
                    window["__IMG_SETTINGS_CB__"].update(disabled=False)

            else:
                window["__PREVIEW_FRAME_SECOND__"].update(disabled=True)
                window["__IMG_SETTINGS_CB__"].update(value=False)
                window["__IMG_SETTINGS__"].update(visible=False)
                window["__IMG_SETTINGS_CB__"].update(disabled=True)


        if event == "__RESTORE_DEFAULT_SETTINGS__":
            correct.blue_level = 0.8
            correct.sharpness_level = 1
            correct.white_balance_level = 0
            correct.adjust_red_level = 1
            correct.adjust_green_level = 1
            correct.adjust_blue_level = 1
            correct.contrast_level = 1
            correct.gamma_level = 1
            correct.brightness_level = 1
            correct.sat_level = 1.0
            correct.cb_level = 1
            correct.denoising_level = 0

            window["__COLOR_BALANCE_CB__"].update(value=False)
            window["__COLOR_BALANCE_SLIDER__"].update(value=correct.cb_level)
            window["__UNDERWATER_RESTORATION_BLUE_LEVEL_CB__"].update(value=False)
            window["__UNDERWATER_RESTORATION_BLUE_LEVEL_SLIDER__"].update(value=correct.blue_level)
            window["__WHITE_BALANCE_CB__"].update(value=False)
            window["__WHITE_BALANCE_SLIDER__"].update(value=correct.white_balance_level)
            window["__SATURATION_CB__"].update(value=False)
            window["__SATURATION_SLIDER__"].update(value=correct.sat_level)
            window["__GAMMA_CB__"].update(value=False)
            window["__GAMMA_SLIDER__"].update(value=correct.gamma_level)
            window["__DENOISING_CB__"].update(value=False)
            window["__DENOISING_SLIDER__"].update(value=correct.denoising_level)
            window["__CONTRAST_CB__"].update(value=False)
            window["__CONTRAST_SLIDER__"].update(value=correct.contrast_level)
            window["__BRIGHTNESS_CB__"].update(value=False)
            window["__BRIGHTNESS_SLIDER__"].update(value=correct.brightness_level)
            window["__SHARPNESS_CB__"].update(value=False)
            window["__SHARPNESS_SLIDER__"].update(value=correct.sharpness_level)
            window["__ADJ_RED_LEVEL_CB__"].update(value=False)
            window["__ADJ_RED_LEVEL_SLIDER__"].update(value=correct.adjust_red_level)
            window["__ADJ_GREEN_LEVEL_CB__"].update(value=False)
            window["__ADJ_GREEN_LEVEL_SLIDER__"].update(value=correct.adjust_green_level)
            window["__ADJ_BLUE_LEVEL_CB__"].update(value=False)
            window["__ADJ_BLUE_LEVEL_SLIDER__"].update(value=correct.adjust_blue_level)

        if (event == "__INPUT_FILE_LIST__" and len(values["__INPUT_FILE_LIST__"]) and event != "__CORRECT__"  and event != "__CORRECT_SINGLE__") or (event == "__REFRESH_PREVIEW__" and len(values["__INPUT_FILE_LIST__"]) and event != "__CORRECT__") or (event == "__INPUT_FILES__") or (event == "__PREVIEW_CB__"):


            if(event == "__REFRESH_PREVIEW__"):
                correct.preview_log = ''
                correct.preview_errors_log = []
                correct.preview_mode = 1

            if ((values["__PREVIEW_CB__"] == True)):
                window["__INPUT_FILE_LIST__"].update(select_mode='SINGLE')

                if event == "__INPUT_FILES__":
                    if(values["__INPUT_FILES__"]!=''):
                        selected_item_path = values["__INPUT_FILES__"].split(";")[0]

                else:
                    # get selected item index
                    if(len(values["__INPUT_FILE_LIST__"])==0):
                        if(len([x for x in window["__INPUT_FILE_LIST__"].get_list_values()])!=0):
                            window["__INPUT_FILE_LIST__"].update(set_to_index=0)
                            selected_item_path = [x for x in window["__INPUT_FILE_LIST__"].get_list_values()][0]

                        else:
                            selected_item_path == ''

                    else:
                        list_box_selected_item = window.Element('__INPUT_FILE_LIST__').Widget.curselection()
                        selected_item_path = values["__INPUT_FILE_LIST__"][0]



                if(selected_item_path != ''):
#####################################################################
                    preview = None
                    preview_before = None
                    preview_after = None

                    extension = str(selected_item_path)[str(selected_item_path).rfind("."):].lower()

                    filename = os.path.basename(str(selected_item_path))

                    window.Element('__IMG_SETTINGS__').Update(visible=False)

                    if extension in IMAGE_TYPES:
                        preview = correct_image(str(selected_item_path),None,None)
                    if extension in VIDEO_TYPES:

                        video_duration=correct.get_video_duration(str(selected_item_path))
                        preview_second = float(values["__PREVIEW_FRAME_SECOND__"])


                        if(int(preview_second)<int(video_duration)):
                            preview = correct_image(None,None,correct.get_video_frame(str(selected_item_path), preview_second))
                        else:
                            preview = correct_image(None, None, correct.get_video_frame(str(selected_item_path), round(int(video_duration)-1)))

                    preview_before = preview[0]
                    preview_after = preview[1]

                    window["__PHOTO_NAME__"].update("Preprocessing of file: "+filename)

                    window["__PREVIEW_BEFORE__"](data=preview_before)
                    window["__PREVIEW_AFTER__"](data=preview_after)

                    window.Element('__VIDEO_VIEWER__').Update(visible=False)
                    window.Element('__PHOTO_VIEWER__').Update(visible=True)

                    #window["__INPUT_FILE_LIST__"].update(set_to_index=list_box_selected_item[0])


                    if (values["__IMG_SETTINGS_CB__"] == True):
                        window["__IMG_SETTINGS__"].update(visible=True)
                    else:
                        if (values["__IMG_SETTINGS_CB__"] == False):
                            window["__IMG_SETTINGS__"].update(visible=False)

                    if (event == "__REFRESH_PREVIEW__"):

                        preview_error_list = []
                        if(len(correct.preview_errors_log)==0):
                            #print("No errors")
                            correct.preview_log = 'No errors. All settings are correct.'
                            preview_error_list.append(correct.preview_log)
                        else:
                            #print("The are several errors: ",len(correct.preview_errors_log))
                            if(len(correct.preview_errors_log)>1):
                                correct.preview_log = 'There are '+ str(len(correct.preview_errors_log)) + ' errors:\n'
                            else:
                                correct.preview_log = 'There is ' + str(len(correct.preview_errors_log)) + ' error:\n'
                            preview_error_list.append(correct.preview_log)
                            for item in correct.preview_errors_log:
                                preview_error_list.append(' --- '+ str(item))

                        window["__PREVIEW_STATUS__"].update(values=preview_error_list)
                        correct.preview_mode = 0



                    window.Refresh()



            else:
                window.Element('__VIDEO_VIEWER__').Update(visible=False)
                window.Element('__PHOTO_VIEWER__').Update(visible=False)

#####################################################################

        if event == "__IMG_SETTINGS_CB__":
            if (values["__IMG_SETTINGS_CB__"] == True):
                window["__IMG_SETTINGS__"].update(visible=True)
            else:
                if (values["__IMG_SETTINGS_CB__"] == False):
                    window["__IMG_SETTINGS__"].update(visible=False)


        if event == "__PREVIEW_PREVIOUS_FRAME__":
            if(int(values["__PREVIEW_FRAME_SECOND__"])>1):
                window["__PREVIEW_FRAME_SECOND__"].update(value=int(values["__PREVIEW_FRAME_SECOND__"])-1)

        if event == "__PREVIEW_NEXT_FRAME__":
            window["__PREVIEW_FRAME_SECOND__"].update(value=int(values["__PREVIEW_FRAME_SECOND__"])+1)

        if event == "__INPUT_FILES__":

            list_items = []
            existing_filepaths = [x for x in window["__INPUT_FILE_LIST__"].get_list_values()]
            filepaths = existing_filepaths + values["__INPUT_FILES__"].split(";")

            for item in filepaths:
                if item not in list_items:
                    list_items.append(item)

            filepaths=list_items


            ################################################

            #########################################################
            # Populate listbox with filenames
            input_filepaths = [f for f in filepaths if valid_file(f)]
            window["__INPUT_FILE_LIST__"].update(input_filepaths)

            # Change output folder to the same as input
            if len(input_filepaths) > 0:
                if(values["__PREVIEW_CB__"] == True):
                    window["__INPUT_FILE_LIST__"].update(set_to_index=len(input_filepaths)-1)
                #if(values["__PREVIEW_CB__"] == True):
                   #window["__PHOTO_VIEWER__"].update(visible=True)
                window["__OUTPUT_FOLDER__"].update(os.path.dirname(input_filepaths[0]))

            if(len([x for x in window["__INPUT_FILE_LIST__"].get_list_values()])!=0):
                window["__IMG_SETTINGS_CB__"].update(disabled=False)

        #--------------------Checkboxes --------------------------------
        if event == "__OUTPUT_FOLDER_CB__":
            if(values["__OUTPUT_FOLDER_CB__"]==False):
                window["__OUTPUT_FOLDER__"].update(disabled=True)
                window["__OUT_FOLDER_BROWSE_BUTTON__"].update(disabled=True)
            else:
                window["__OUTPUT_FOLDER__"].update(disabled=False)
                window["__OUT_FOLDER_BROWSE_BUTTON__"].update(disabled=False)
                window["__OUTPUT_PREFIX__"].update(disabled=True)

        if event == "__UNDERWATER_RESTORATION_BLUE_LEVEL_CB__":
            if (values["__UNDERWATER_RESTORATION_BLUE_LEVEL_CB__"] == True):
                window["__UNDERWATER_RESTORATION_BLUE_LEVEL_SLIDER__"].update(disabled=False)
            else:
                window["__UNDERWATER_RESTORATION_BLUE_LEVEL_SLIDER__"].update(disabled=True)
        if event == "__UNDERWATER_RESTORATION_BLUE_LEVEL_SLIDER__":
            if(float(values["__UNDERWATER_RESTORATION_BLUE_LEVEL_SLIDER__"])!=correct.blue_level):
                correct.blue_level = values["__UNDERWATER_RESTORATION_BLUE_LEVEL_SLIDER__"]

        if event == "__SATURATION_CB__":
            if (values["__SATURATION_CB__"] == True):
                window["__SATURATION_SLIDER__"].update(disabled=False)
            else:
                window["__SATURATION_SLIDER__"].update(disabled=True)
        if event == "__SATURATION_SLIDER__":
            if(values["__SATURATION_SLIDER__"]!=correct.sat_level):
                correct.sat_level = float(values["__SATURATION_SLIDER__"])

        if event == "__GAMMA_CB__":
            if (values["__GAMMA_CB__"] == True):
                window["__GAMMA_SLIDER__"].update(disabled=False)
            else:
                window["__GAMMA_SLIDER__"].update(disabled=True)
        if event == "__GAMMA_SLIDER__":
            if(values["__GAMMA_SLIDER__"]!=correct.gamma_level):
                correct.gamma_level = float(values["__GAMMA_SLIDER__"])

        if event == "__DENOISING_CB__":
            if (values["__DENOISING_CB__"] == True):
                window["__DENOISING_SLIDER__"].update(disabled=False)
            else:
                window["__DENOISING_SLIDER__"].update(disabled=True)
        if event == "__DENOISING_SLIDER__":
            if(values["__DENOISING_SLIDER__"]!=correct.denoising_level):
                correct.denoising_level = float(values["__DENOISING_SLIDER__"])

        if event == "__CONTRAST_CB__":
            if (values["__CONTRAST_CB__"] == True):
                window["__CONTRAST_SLIDER__"].update(disabled=False)
            else:
                window["__CONTRAST_SLIDER__"].update(disabled=True)
        if event == "__CONTRAST_SLIDER__":
            if(values["__CONTRAST_SLIDER__"]!=correct.contrast_level):
                correct.contrast_level = float(values["__CONTRAST_SLIDER__"])

        if event == "__BRIGHTNESS_CB__":
            if (values["__BRIGHTNESS_CB__"] == True):
                window["__BRIGHTNESS_SLIDER__"].update(disabled=False)
            else:
                window["__BRIGHTNESS_SLIDER__"].update(disabled=True)
        if event == "__BRIGHTNESS_SLIDER__":
            if(values["__BRIGHTNESS_SLIDER__"]!=correct.contrast_level):
                correct.brightness_level = float(values["__BRIGHTNESS_SLIDER__"])

        if event == "__SHARPNESS_CB__":
            if (values["__SHARPNESS_CB__"] == True):
                window["__SHARPNESS_SLIDER__"].update(disabled=False)
            else:
                window["__SHARPNESS_SLIDER__"].update(disabled=True)
        if event == "__SHARPNESS_SLIDER__":
            if(values["__SHARPNESS_SLIDER__"]!=correct.contrast_level):
                correct.sharpness_level = float(values["__SHARPNESS_SLIDER__"])

########################################
        if event == "__ADJ_RED_LEVEL_CB__":
            if (values["__ADJ_RED_LEVEL_CB__"] == True):
                window["__ADJ_RED_LEVEL_SLIDER__"].update(disabled=False)
            else:
                window["__ADJ_RED_LEVEL_SLIDER__"].update(disabled=True)
        if event == "__ADJ_RED_LEVEL_SLIDER__":
            if (values["__ADJ_RED_LEVEL_SLIDER__"] != correct.adjust_red_level):
                correct.adjust_red_level = float(values["__ADJ_RED_LEVEL_SLIDER__"])

        if event == "__ADJ_GREEN_LEVEL_CB__":
            if (values["__ADJ_GREEN_LEVEL_CB__"] == True):
                window["__ADJ_GREEN_LEVEL_SLIDER__"].update(disabled=False)
            else:
                window["__ADJ_GREEN_LEVEL_SLIDER__"].update(disabled=True)
        if event == "__ADJ_GREEN_LEVEL_SLIDER__":
            if (values["__ADJ_GREEN_LEVEL_SLIDER__"] != correct.adjust_green_level):
                correct.adjust_green_level = float(values["__ADJ_GREEN_LEVEL_SLIDER__"])

        if event == "__ADJ_BLUE_LEVEL_CB__":
            if (values["__ADJ_BLUE_LEVEL_CB__"] == True):
                window["__ADJ_BLUE_LEVEL_SLIDER__"].update(disabled=False)
            else:
                window["__ADJ_BLUE_LEVEL_SLIDER__"].update(disabled=True)
        if event == "__ADJ_BLUE_LEVEL_SLIDER__":
            if (values["__ADJ_BLUE_LEVEL_SLIDER__"] != correct.adjust_blue_level):
                correct.adjust_blue_level = float(values["__ADJ_BLUE_LEVEL_SLIDER__"])


#######################################
        if event == "__WHITE_BALANCE_CB__":
            if (values["__WHITE_BALANCE_CB__"] == True):
                window["__WHITE_BALANCE_SLIDER__"].update(disabled=False)
            else:
                window["__WHITE_BALANCE_SLIDER__"].update(disabled=True)

        if event == "__WHITE_BALANCE_SLIDER__":
            if (values["__WHITE_BALANCE_SLIDER__"] != correct.white_balance_level):
                correct.white_balance_level = float(values["__WHITE_BALANCE_SLIDER__"])

        if event == "__COLOR_BALANCE_CB__":
            if(values["__COLOR_BALANCE_CB__"] == True):
                window["__COLOR_BALANCE_SLIDER__"].update(disabled=False)
            else:
                window["__COLOR_BALANCE_SLIDER__"].update(disabled=True)
        if event == "__COLOR_BALANCE_SLIDER__":
            if(values["__COLOR_BALANCE_SLIDER__"]!=correct.cb_level):
                correct.cb_level = float(values["__COLOR_BALANCE_SLIDER__"])

        if event == "__TEMP_FOLDER_CB__":
            if (values["__TEMP_FOLDER_CB__"] == False):
                window["__TEMP_FOLDER__"].update(disabled=True)
                window["__TEMP_FOLDER_BROWSE_BUTTON__"].update(disabled=True)
                window["__TEMP_FOLDER__"].update(value=TempDirPath)

            else:
                window["__TEMP_FOLDER__"].update(disabled=False)
                window["__TEMP_FOLDER_BROWSE_BUTTON__"].update(disabled=False)

        if event == "__OUTPUT_PREFIX_CB__":

            if(values["__OUTPUT_PREFIX_CB__"]==False):
                window["__OUTPUT_PREFIX__"].update(disabled=True)
            else:
                window["__OUTPUT_PREFIX__"].update(disabled=False)
        #---------------------------------------------------------------

        if (event == "__OUTPUT_FOLDER__" and values["__OUTPUT_FOLDER_CB__"]==True):
            window["__OUTPUT_FOLDER__"].update(values["__OUTPUT_FOLDER__"])

        if (event == "__CORRECT__") or (event == "__CORRECT_SINGLE__"):

            if (event == "__CORRECT__"):
                filepaths = [x for x in window["__INPUT_FILE_LIST__"].get_list_values()]
                window["__IMG_SETTINGS__"].update(visible=False)
                # clear preview listbox
                window["__PREVIEW_STATUS__"].update(values='')
            else:
                if(event == "__CORRECT_SINGLE__"):
                    # clear preview listbox
                    window["__PREVIEW_STATUS__"].update(values='')
                    filepaths  = values["__INPUT_FILE_LIST__"]
                    window["__IMG_SETTINGS__"].update(visible=False)

            file_generator = get_files(filepaths)

            window["__CORRECT__"].update(disabled=True)
            window["__CANCEL__"].update(disabled=False)
            window["__CLEAR_LIST__"].update(disabled=True)


            window["__PREVIEW_CB__"].update(disabled=True)
            window["__PREVIEW_FRAME_SECOND__"].update(disabled=True)
            window["__REFRESH_PREVIEW__"].update(disabled=True)

            #window["__COLOR_BALANCE_CB__"].update(disabled=True)

            #window["__SATURATION_CB__"].update(disabled=True)

            #window["__UNDERWATER_RESTORATION_BLUE_LEVEL_CB__"].update(disabled=True)

        if event == "__CANCEL__":
            window["__IMG_SETTINGS__"].update(visible=False)
            window["__CORRECT__"].update(disabled=False)
            window["__CANCEL__"].update(disabled=False)
            window["__CLEAR_LIST__"].update(disabled=False)
            file_generator = None
            file_index = 0
            analyze_video_generator = None
            process_video_generator = None

            #window["__INPUT_FILE_LIST__"].update(disabled=False)
            window["__PREVIEW_CB__"].update(disabled=False)
            window["__PREVIEW_FRAME_SECOND__"].update(disabled=False)
            window["__REFRESH_PREVIEW__"].update(disabled=False)

            window["__STATUS__"].update("Cancelled")

            window["__COLOR_BALANCE_CB__"].update(disabled=False)
            #window["__COLOR_BALANCE_SLIDER__"].update(disabled=False)
            window["__SATURATION_CB__"].update(disabled=False)
            #window["__SATURATION_SLIDER__"].update(disabled=False)
            window["__UNDERWATER_RESTORATION_BLUE_LEVEL_CB__"].update(disabled=False)
            #window["__UNDERWATER_RESTORATION_BLUE_LEVEL_SLIDER__"].update(disabled=False)


        if event == "__CLEAR_LIST__":
            window["__IMG_SETTINGS__"].update(visible=False)
            window["__INPUT_FILE_LIST__"].update(values=[])
            window["__STATUS__"].update("")
            window.Element('__VIDEO_VIEWER__').Update(visible=False)
            window.Element('__PHOTO_VIEWER__').Update(visible=False)

            window["__COLOR_BALANCE_CB__"].update(disabled=False)

            window["__IMG_SETTINGS_CB__"].update(disabled=True)

            #window["__COLOR_BALANCE_SLIDER__"].update(disabled=False)
            window["__SATURATION_CB__"].update(disabled=False)
            #window["__SATURATION_SLIDER__"].update(disabled=False)
            window["__UNDERWATER_RESTORATION_BLUE_LEVEL_CB__"].update(disabled=False)
            #window["__UNDERWATER_RESTORATION_BLUE_LEVEL_SLIDER__"].update(disabled=False)



        if analyze_video_generator:
            try:
                # need to set i null before video processing
                item = next(analyze_video_generator)
                if type(item) == dict:
                    video_data = item
                    process_video_generator = process_video(video_data, True)
                    analyze_video_generator = None
                elif type(item) == int:
                    count = item
                    status_message = f"Analyzing: {count} frames"
                    window["__STATUS__"].update(status_message)
                else:
                    pass

            except StopIteration:
                window["__STATUS__"].update("Analysis done")
                analyze_video_generator = None

            except:
                window["__STATUS__"].update("Analysis failed")
                analyze_video_generator = None

            continue

        if process_video_generator:
            try:

                window["__IMG_SETTINGS__"].update(visible=False)
                if (cl.video_progress_percentage > 99):
                    cl.video_progress_percentage = 100
                if (cl.audio_progress_percentage > 99):
                    cl.audio_progress_percentage = 100
                # need to update before processing video

                window["__SOUND_PROGBAR__"].UpdateBar(round(cl.video_progress_percentage))
                window["__SOUND_PROGBAR_PERCENTS__"].update(str(round(cl.video_progress_percentage)) + "%")

                window["__SOUND_EX_PROGBAR__"].UpdateBar(round(cl.audio_progress_percentage))
                window["__SOUND_EX_PROGBAR_PERCENTS__"].update(str(round(cl.audio_progress_percentage)) + "%")

                percent, preview = next(process_video_generator)
                window.Element('__VIDEO_VIEWER__').Update(visible=True)

                window["__VIDEO_PREVIEW__"](data=preview)

                status_message = "Color restoration processing: {:} %".format(round(percent))

                window["__STATUS__"].update(status_message)

                window.Element('__PHOTO_VIEWER__').Update(visible=False)

                if(percent>99):
                    percent=100
                window["__PROGBAR_PERCENTS__"].update(str(round(percent)) + "%")
                window["__PROGBAR__"].UpdateBar(round(percent))
                window["__VIDEO_NAME__"].update("Processing of file: "+current_in_filename)


            except StopIteration:

                if (cl.audio_progress_percentage > 99):
                    cl.audio_progress_percentage = 100
                if (cl.video_progress_percentage > 99):
                    cl.video_progress_percentage = 100
                window["__SOUND_PROGBAR__"].UpdateBar(round(cl.video_progress_percentage))
                window["__SOUND_PROGBAR_PERCENTS__"].update(str(round(cl.video_progress_percentage)) + "%")

                window["__SOUND_EX_PROGBAR__"].UpdateBar(round(cl.audio_progress_percentage))
                window["__SOUND_EX_PROGBAR_PERCENTS__"].update(str(round(cl.audio_progress_percentage)) + "%")



                if(cl.audio_progress_percentage>0 and cl.video_progress_percentage==0.0):
                    window["__STATUS__"].update("Audio extraction processing: {:} %".format(round(cl.audio_progress_percentage)))
                if (cl.audio_progress_percentage == 100 and cl.video_progress_percentage <100):
                    window["__STATUS__"].update("Video encoding processing: {:} %".format(round(cl.video_progress_percentage)))


                if(round(cl.audio_progress_percentage)==100 and round(cl.video_progress_percentage)==100):
                    window["__STATUS__"].update("Processing done")
                    process_video_generator = None


            except:
                window["__STATUS__"].update("Processing failed")
                process_video_generator = None
                window.Element('__VIDEO_VIEWER__').Update(visible=False)

            continue

        if file_generator:
            correct.temp_dir_path = values["__TEMP_FOLDER__"]
            if(os.path.exists(correct.temp_dir_path)!=True):
                try:
                    os.mkdir(correct.temp_dir_path)
                except:
                    print("Can't create ", correct.temp_dir_path," "+"folder.")

            try:
                if (cl.audio_progress_percentage == 0.0 and cl.video_progress_percentage == 0.0):
                    f = next(file_generator)
                    listbox_hight_rows = list_of_files_size


                    if(file_index%listbox_hight_rows==0):
                        window["__INPUT_FILE_LIST__"].update(scroll_to_index = file_index)
                    file_index += 1
                    current_in_filename=os.path.basename(f)

                    if(event == "__CORRECT_SINGLE__"):
                        window["__INPUT_FILE_LIST__"].update(set_to_index = list_box_selected_item)
                    else:
                        window["__INPUT_FILE_LIST__"].update(set_to_index=file_index-1)

                    if(values["__OUTPUT_PREFIX_CB__"] == True):
                       new_filename = values["__OUTPUT_PREFIX__"] + "_" + os.path.basename(f)
                    else:
                       new_filename=os.path.basename(f)


                    if (values["__OUTPUT_FOLDER_CB__"] == True):
                        output_filepath = os.path.join(values["__OUTPUT_FOLDER__"], new_filename)
                    else:
                        output_filepath = os.path.dirname(f)+"/"+new_filename
                    ############################################################################

                    extension = f[f.rfind("."):].lower()

                    if extension in IMAGE_TYPES:

                        preview = None
                        preview_before = None
                        preview_after = None

                        preview = correct_image(f, output_filepath,None)

                        preview_before = preview[0]
                        preview_after = preview[1]

                        window["__PHOTO_NAME__"].update("Processing of file: "+current_in_filename)

                        window["__PREVIEW_BEFORE__"](data=preview_before)
                        window["__PREVIEW_AFTER__"](data=preview_after)

                        window.Element('__VIDEO_VIEWER__').Update(visible=False)
                        window.Element('__PHOTO_VIEWER__').Update(visible=True)

                        if (values["__IMG_SETTINGS_CB__"] == True):
                            window["__IMG_SETTINGS__"].update(visible=True)
                        else:
                            if (values["__IMG_SETTINGS_CB__"] == False):
                                window["__IMG_SETTINGS__"].update(visible=False)

                        window.Refresh()

                    if extension in VIDEO_TYPES:

                        window["__STATUS__"].update("Analyzing")
                        analyze_video_generator = analyze_video(f, output_filepath)

            except StopIteration:
                window["__STATUS__"].update("All done!")
                window["__CORRECT__"].update(disabled=False)
                window["__CLEAR_LIST__"].update(disabled=False)
                #window.Element('__VIDEO_VIEWER__').Update(visible=False)
                #window.Element('__PHOTO_VIEWER__').Update(visible=False)
                file_generator = None
                file_index = 0
                analyze_video_generator = None
                process_video_generator = None

                #window["__INPUT_FILE_LIST__"].update(disabled=False)
                window["__PREVIEW_CB__"].update(disabled=False)
                window["__PREVIEW_FRAME_SECOND__"].update(disabled=False)
                window["__REFRESH_PREVIEW__"].update(disabled=False)

                window["__COLOR_BALANCE_CB__"].update(disabled=False)
                #window["__COLOR_BALANCE_SLIDER__"].update(disabled=False)
                window["__SATURATION_CB__"].update(disabled=False)
                #window["__SATURATION_SLIDER__"].update(disabled=False)
                window["__UNDERWATER_RESTORATION_BLUE_LEVEL_CB__"].update(disabled=False)
                #window["__UNDERWATER_RESTORATION_BLUE_LEVEL_SLIDER__"].update(disabled=False)

            except:
                window["__STATUS__"].update("Error in accessing file")
                window["__CORRECT__"].update(disabled=False)
                window["__CLEAR_LIST__"].update(disabled=False)

                #window["__INPUT_FILE_LIST__"].update(disabled=False)
                window["__PREVIEW_CB__"].update(disabled=False)
                window["__PREVIEW_FRAME_SECOND__"].update(disabled=False)
                window["__REFRESH_PREVIEW__"].update(disabled=False)

                file_generator = None
                file_index = 0
                analyze_video_generator = None
                process_video_generator = None