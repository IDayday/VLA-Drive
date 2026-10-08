from shapely import affinity
from shapely.geometry import MultiPolygon
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer

class VADPolygonMap:
    def __init__(self, dataroot, location):
        self.map_explorer = {location: NuScenesMapExplorer(NuScenesMap(dataroot=dataroot, map_name=location))}

    def get_contour_line(self,patch_box,patch_angle,layer_name,location):
            if layer_name not in self.map_explorer[location].map_api.non_geometric_polygon_layers:
                raise ValueError('{} is not a polygonal layer'.format(layer_name))
    
            patch_x = patch_box[0]
            patch_y = patch_box[1]
    
            patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)
    
            records = getattr(self.map_explorer[location].map_api, layer_name)
    
            polygon_list = []
            if layer_name == 'drivable_area':
                for record in records:
                    polygons = [self.map_explorer[location].map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]
    
                    for polygon in polygons:
                        new_polygon = polygon.intersection(patch)
                        if not new_polygon.is_empty:
                            new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                          origin=(patch_x, patch_y), use_radians=False)
                            new_polygon = affinity.affine_transform(new_polygon,
                                                                    [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                            if new_polygon.geom_type == 'Polygon':
                                new_polygon = MultiPolygon([new_polygon])
                            polygon_list.append(new_polygon)
    
            else:
                for record in records:
                    polygon = self.map_explorer[location].map_api.extract_polygon(record['polygon_token'])
    
                    if polygon.is_valid:
                        new_polygon = polygon.intersection(patch)
                        if not new_polygon.is_empty:
                            new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                          origin=(patch_x, patch_y), use_radians=False)
                            new_polygon = affinity.affine_transform(new_polygon,
                                                                    [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                            if new_polygon.geom_type == 'Polygon':
                                new_polygon = MultiPolygon([new_polygon])
                            polygon_list.append(new_polygon)
    
            return polygon_list
