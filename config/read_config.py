import yaml

def config():
    with open('./config/config.yaml', 'r') as f:
        data = yaml.safe_load(f)
    return data

if __name__ == '__main__':
    print(config())