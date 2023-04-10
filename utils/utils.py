def download_image_by_url(image_url):
    with urllib.request.urlopen(image_url) as url:
        img = Image.open(url)
        url.close()
    return img

def load_image(path):
    return Image.open(path)

def get_image_path_by_id(root, image_id):
    path = Path(root) / data.loc[image_id]['file_name']
    path.with_suffix('.jpg')
    
def load_image_by_id(data, image_id, compressed=True):
    file_name = data.loc[image_id]['file_name']
    if compressed:
        file_name = change_file_ext(file_name)
    
    return load_image(file_name, compressed)

def draw_rectangle(image, xy, color='red'):
    img = ImageDraw.Draw(image) 
    img.rectangle(xy, outline=color, width=3)
    return image

def create_bboxes(annotation_data):
    df = annotation_data.copy()
    df['bbox'] = df['bbox'].apply(ast.literal_eval)

    groupby_image = df.groupby(by='image_id')
    df = groupby_image['bbox'].apply(list).reset_index(name='bboxes').set_index('image_id')
    df['category_ids'] = groupby_image['category_id'].apply(list)  
    return df

def create_yolo_dataset(
    image_data,
    annotation_data,
    data_type='train', 
    image_root=Cfg.IMAGES_ROOT, 
    dataset_root=Cfg.DATASET_ROOT
):
    bboxes_data = create_bboxes(annotation_data)
    image_ids = image_data.index
    
    for image_id in tqdm(image_ids, total=len(image_ids)):
        bounding_bboxes = bboxes_data['bboxes'].loc[image_id]
        category_ids = bboxes_data['category_ids'].loc[image_id]

        image_row = image_data.loc[image_id]
        image_width = image_row['width']
        image_height = image_row['height']
        
        file_name = Path(image_row['file_name']).with_suffix('.jpg')
        source_image_path = Cfg.TRAIN_IMAGES_ROOT / file_name
        target_image_path = dataset_root / f'images/{data_type}/{file_name}'
        label_path = (dataset_root / f'labels/{data_type}/{file_name}').with_suffix('.txt')
        
        #print(file_name)
        
        yolo_data = []
        for bbox, category in zip(bounding_bboxes, category_ids):
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            x_center = x + w/2
            y_center = y + h/2
            x_center /= image_width
            y_center /= image_height
            w /= image_width
            h /= image_height
            
            yolo_data.append([category, x_center, y_center, w, h])

        yolo_data = np.array(yolo_data)

        # Create YOLO lable file
        np.savetxt(label_path, yolo_data, fmt=["%d", "%f", "%f", "%f", "%f"])

        # Copy image file
        shutil.copy(source_image_path, target_image_path)