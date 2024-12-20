
class annotation():
    def __init__(self,annot_str):

        self.main_image="-1.jpg"
        self.action="r_spike"
        self.players=[]
        self.extract_features_from_str(annot_str)
    def extract_features_from_str(self,str):
        features=str.split()
        self.main_image=features[0]
        self.action=features[1]
        for i in range(2,len(features),5):
            cur_player=player(features[i],features[i+1],features[i+2],features[i+3],features[i+4])
            self.players.append(cur_player)
        



class player():
    def __init__(self,top_left_x,top_left_y,bottom_right_x,bottom_right_y,action):
        self.x1=top_left_x
        self.y1=top_left_y
        self.x2=bottom_right_x
        self.y2=bottom_right_y
        self.action=action
    
