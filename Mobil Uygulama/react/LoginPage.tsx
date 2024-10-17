
import {View,Text, Pressable,ImageBackground} from "react-native"

const Login = (navigator) => {
    return (
        <ImageBackground source={require("./sa.jpg")}>
        <View className="h-screen flex justify-center items-center relative" >
            <Text className="text-white absolute top-20 font-ArchivoBlack_Regular text-2xl" > Hoşgeldin Eren Kına </Text>
            <Pressable className=" rounded-md p-5 bg-blue-600 flex items-center justify-center"  onPress={()=>{navigator.navigation.navigate("userpage")}} style={{width:"90%", height:"8%"}}>
                <Text className="text-white font-ArchivoBlack_Regular text-xl " >
                    Giriş Yap
                </Text>
            </Pressable>
        </View>
        </ImageBackground>
        
    );
}
 
export default Login;