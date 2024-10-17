import { Text, View, Image, Pressable , ImageBackground} from "react-native";
import LinearGradient from "react-native-linear-gradient";
const Protez = (navigator) => {
    return (
    <ImageBackground source={require("./sa.jpg")}>
        <View className="flex h-screen">
        
        <View className=" flex items-center justify-center align-middle" style={{backgroundColor:"#06090F", height:60}}>
        <Pressable className="absolute top-2 left-1.5" onPress={()=>{navigator.navigation.navigate("userpage")}}>
        <Image source={require("./logo.png")} style={{
              margin:4,
              resizeMode: 'cover',
              height: 40,
              width: 40,
            }} className=""></Image></Pressable>
        <View className="flex items-center p-1 m-1 justify-center align-middle">

<Text className="text-white text-xl top-2 pb-2 font-ArchivoBlack_Regular" style={{height:40}}>Eren Kına'nın Protez Analizi
</Text>
</View>
</View>
      <LinearGradient colors={["#06090F","#060910"]} className="h-5"></LinearGradient>
      <View className="flex flex-grow items-center justify-center bottom-10" >
        <View>
        <Text className="text-blue-600 text-xl font-ArchivoBlack_Regular text-center">Olası Protez Sorunları : {"\n"}</Text>
        <Text className="text-white text-xl font-ArchivoBlack_Regular text-center">Sol Ayak Bileğinde Sorun Tespit Edilmiştir</Text>
        </View>
      </View>
     
        </View>
        </ImageBackground>
    );
}
 
export default Protez;